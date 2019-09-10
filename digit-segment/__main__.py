#!/usr/bin/env python

"""
Digit Segmenter
"""
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from Helper import get_opts

usage = "python digit-segment/ --img <images wildcard> --out <output directory>"
commands = ["img", "out"]
debug = False


def remove_hour_mark(image, hour_mask):
    return cv2.add(image, hour_mask)


def get_histogram(image):
    return np.sum(cv2.bitwise_not(image), axis=0).tolist()


def try_mser(gray):
    # possibly erode/dilate if seeing any issue.
    #kernel2 = np.ones((1, 1), np.uint8)
    #dilation = cv2.dilate(blurred, kernel2, iterations=2)
    #cv2.imshow("AfterDilation", dilation)
    #cv2.waitKey(0)

    #kernel1 = np.ones((2, 2), np.uint8)
    #erosion = cv2.erode(dilation, kernel1, iterations=2)
    #cv2.imshow("AfterErosion", erosion)
    #cv2.waitKey(0)

    # Using MSER Object detection
    mser = cv2.MSER_create()
    regions, br = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    # Draw MSER detected areas
    vis = gray.copy()
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    if debug:
        cv2.imshow('detected hulls', vis)
        cv2.waitKey(0)

    # Show with text only?
    mask = np.zeros((gray.shape[0], gray.shape[1], 1), dtype=np.uint8)
    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    # this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(gray, gray, mask=mask)

    if debug:
        cv2.imshow("text only", text_only)
        cv2.waitKey(0)

    # Create Digit Images.
    valid_rect = []
    overlap_perc_for_similar_threshold = 0.8
    for rect in br:
        if not is_potential_digit(rect):
            continue
        if exists_similar(rect, valid_rect, overlap_perc_for_similar_threshold):
            continue
        valid_rect.append(rect.copy())

    # Show Valid Images.
    vimg = []
    for rect in valid_rect:
        x, y, w, h = rect
        vimg.append(gray[y:y + h, x:x + w].copy())
        cv2.waitKey(0)

    return vimg


def is_potential_digit(rect):
    x, y, w, h = rect
    return h > 3 and w > 3  # Complete heuristic to remove black around image.


def exists_similar(rect, vrects, area_overlap):
    return any(get_overlap(rect, valid_rect) > area_overlap for valid_rect in vrects)


def get_overlap(lhs, rhs):
    x1, y1, w1, h1 = lhs
    x2, y2, w2, h2 = rhs

    # Some stackoverflow code to compute overlap of rectangle A and B. Seem to come from VB, ugly but meh.
    SA = w1 * h1
    SB = w2 * h2
    SI = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    SU = SA + SB - SI
    perc_overlap = SI / SU

    return perc_overlap


def make_similar_to_mnist(digits_img):
    # STEP 4.5

    # These steps process the scanned images to be in the same format and have the same properties as the EMNIST images
    # They are described by the EMNIST authors in detail here: https://arxiv.org/abs/1702.05373v1
    processed_story = []

    for crop_img in digits_img:
        height, width = crop_img.shape[:2]
        max_side = max(height, width) + 5

        canvas = np.zeros((max_side, max_side), dtype=np.uint8)
        canvas.fill(255)

        off_x = int((max_side / 2) - (width / 2))
        off_y = int((max_side / 2) - (height / 2))

        canvas[off_y: off_y + height, off_x: off_x + width] = crop_img

        img = canvas

        # step 1: Apply Gaussian blur filter
        img = cv2.GaussianBlur(img, (7, 7), 0)

        # step 4: Resize and resample to be 28 x 28 pixels
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)

        # step 5: Normalize pixels and reshape before adding to the new story array
        img = img / 255
        img = img.reshape((28, 28))
        processed_story.append(img)

    return np.hstack(processed_story).copy()


def process(image, hour_mask):

    # Load and show original image
    if debug:
        cv2.imshow("original-image", image)
        cv2.waitKey(0)

    # Show image no hour mark
    image_no_hour = remove_hour_mark(image, hour_mask)
    if debug:
        cv2.imshow("Hour no marks", image_no_hour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Show histogram, or separate digits
    if debug:
        histogram = get_histogram(image_no_hour.copy())
        plt.plot(histogram)
        plt.show()

    raw_digit_imgs = try_mser(image_no_hour.copy())
    for digit_img in raw_digit_imgs:
        if debug:
            cv2.imshow("digit", digit_img)
            cv2.waitKey(0)

    raw_mnist_line = make_similar_to_mnist(raw_digit_imgs)

    cv2.imshow("res.png", raw_mnist_line)
    cv2.imwrite("res.png", 255 - (255 * raw_mnist_line))
    cv2.waitKey(0)
    #cv2.imwrite("end-result.png", cv2.cvtColor(raw_mnist_line, cv2.COLOR_BGR2GRAY))
    print("Processed the scanned images.")


if __name__ == '__main__':
    image_wildcard, out_dir = get_opts(commands, usage)

    images = glob.glob(image_wildcard)

    hour_mark_mask = cv2.imread("hour_mask.png", cv2.IMREAD_GRAYSCALE)

    print("Starting Digit Segmenter with configuration: %s, %s, %s" % (image_wildcard, images, out_dir))

    for img in images:
        process(cv2.imread(img, cv2.IMREAD_GRAYSCALE), cv2.bitwise_not(hour_mark_mask))  # bitwise not does invert
