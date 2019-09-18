import cv2

import glob
import os

import cv2
from tqdm import tqdm
import numpy as np

from helper import get_opts, as_grayscale
#  ../../clean-cropped
usage = "python cleaning/center.py --img <wildcard> --out <dir>"
commands = ["img", "out"]
threshold_area = 0.3  # in pixels? heuristic
final_size = (176, 97)
debug_vis = False
canny_thresh = 200
dilate_kernel = np.ones((2, 2), np.uint8)
dilate_iter = 1
pad = 4


def keep_contours(contour):
    max_area = (final_size[0] - 2 * pad) * (final_size[1] - 2 * pad)
    contour_area = cv2.contourArea(contour)
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect
    aspect = w / float(h)
    total_area = w * h
    ratio = contour_area / float(total_area)
    if debug_vis:
        print("Aspect Ratio (w%s/h%s) aspect:%.3s contour_area:%s total_area:%s ratio:%s" % (w, h, aspect, contour_area,
                                                                                             total_area, ratio))
    return threshold_area < contour_area < max_area and ratio > 0.00085 and 0.05 < aspect < 20


def process(filename):
    img = as_grayscale(filename)

    tc = img[pad:pad + 97-2*pad, pad:pad + 176 - 2 * pad]

    dilation = cv2.dilate(tc, dilate_kernel, iterations=dilate_iter)

    # Make white on black
    inv = (255 - dilation)

    # Edge detected
    canny_output = cv2.Canny(inv, canny_thresh, canny_thresh * 2)
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find all contours which are not of trivially small area (i.e. not garbage around the border)
    cnts = [cnt for cnt in contours if keep_contours(cnt)]

    if debug_vis:
        img_c = img.copy()
        img_c = cv2.cvtColor(img_c, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img_c, cnts, -1, (0, 255, 0), 3)
        cv2.imshow("1", img_c)
        cv2.waitKey(0)

    if len(cnts) == 0:
        return None

    # Bounding rectangle for all joint points, from all valid contours
    x, y, w, h = cv2.boundingRect(np.vstack(cnts).squeeze())
    # Crop the Region of Interest (ROI)
    cropped = inv[y:y+h+2*pad, x:x+w+2*pad]
    # Return to the original dimensions
    return cv2.resize(cropped, final_size, interpolation=cv2.INTER_CUBIC)


if __name__ == '__main__':
    image_wildcard, out_dir = get_opts(commands, usage)

    print("Stated focusing with images from %s to %s" % (image_wildcard, out_dir))

    assert image_wildcard is not None, "Missing required image wildcard"
    assert out_dir is not None, "Missing required output directory"
    assert os.path.exists(out_dir), "Output directory does not exist."

    for file in tqdm(glob.glob(image_wildcard)):
        img = process(file)
        if img is None:
            continue
        else:
            cv2.imwrite("%s/%s" % (out_dir, os.path.basename(file)), img)
