#!/usr/bin/env python

"""
Digit Segmenter
"""
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Helper import get_opts

usage = "python digit-segment/ --img <images wildcard> --out <output directory>"
commands = ["img", "out"]


def remove_hour_mark(image, hour_mask):
    return cv2.add(image, hour_mask)


def get_histogram(image):
    return np.sum(cv2.bitwise_not(image), axis=0).tolist()


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def get_blobs(image):
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imshow("blob-image", image)
    cv2.waitKey(0)

def try_canny(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    kernel2 = np.ones((1, 1), np.uint8)
    dilation = cv2.dilate(blurred, kernel2, iterations=2)
    cv2.imshow("AfterDilation", dilation)
    cv2.waitKey(0)

    kernel1 = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(dilation, kernel1, iterations=2)
    cv2.imshow("AfterErosion", erosion)
    cv2.waitKey(0)



    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(erosion, 10, 200)
    tight = cv2.Canny(erosion, 225, 250)
    auto = auto_canny(erosion)

    # show the images
    cv2.imshow("Edges", np.hstack([wide, tight, auto]))
    cv2.waitKey(0)

def try_mser(gray):
    # possibly erode/dilate if seeing any issue.

    # Using MSER Object detection
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    # Draw MSER detected areas
    vis = gray.copy()
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    cv2.imshow('detected hulls', vis)
    cv2.waitKey(0)

    # Show with text only?
    mask = np.zeros((gray.shape[0], gray.shape[1], 1), dtype=np.uint8)
    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    # this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(gray, gray, mask=mask)
    cv2.imshow("text only", text_only)
    cv2.waitKey(0)

    for i, contour in enumerate(hulls):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.imshow('test.png', gray[y:y + h, x:x + w])
        cv2.waitKey(0)


def process(image, hour_mask):

    # Load and show original image
    cv2.imshow("original-image", image)
    cv2.waitKey(0)

    # Show image no hour mark
    image_no_hour = remove_hour_mark(image, hour_mask)
    cv2.imshow("Hour no marks", image_no_hour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Show histogram, or separate digits
    #histogram = get_histogram(image_no_hour.copy())
    #plt.plot(histogram)
    #plt.show()

    # Set up the detector with default parameters.
    # All of them, copies since they modify the image.
    #get_blobs(image_no_hour.copy())

    #try_canny(image_no_hour.copy())

    try_mser(image_no_hour.copy())


if __name__ == '__main__':
    image_wildcard, out_dir = get_opts(commands, usage)

    images = glob.glob(image_wildcard)

    hour_mark_mask = cv2.imread("hour_mask.png", cv2.IMREAD_GRAYSCALE)

    print("Starting Digit Segmenter with configuration: %s, %s, %s" % (image_wildcard, images, out_dir))

    for img in images:
        process(cv2.imread(img, cv2.IMREAD_GRAYSCALE), cv2.bitwise_not(hour_mark_mask))  # bitwise not does invert
