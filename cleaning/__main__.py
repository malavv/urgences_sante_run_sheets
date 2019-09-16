#!/usr/bin/env python

"""
Digit Segmenter
"""
import glob
import os

import cv2
from tqdm import tqdm
import numpy as np

from helper import get_opts, as_grayscale

usage = "python cleaning/ --img <wildcard> --out <dir>"
commands = ["img", "out"]

v_black_thresh = 0.95  # Percentage of max blackness of vertical pixels to be considered a border
h_black_thresh = 0.85  # Percentage of max blackness of horizontal pixels to be considered a border
white = 255  # Grayscale white
blank_thresh = 245  # After removing border, taken from a sample of ~10000 images
single_hour_mark = np.zeros((9, 9), dtype=np.uint8)  # Top or Bottom dot of the hour mark


def is_blank(image):
    return np.mean(image) > blank_thresh  # Apply after removing border, because border darken the image


def remove_border(image):
    image = image.copy()
    img_h, img_w = image.shape

    v_means = np.mean(image, axis=0)  # mean for each vertical lines
    h_means = np.mean(image, axis=1)  # mean for each horizontal lines

    # Remove top and bottom border
    for y, h_mean in enumerate(h_means):
        if 10 < y < (img_h - 10):  # Only consider the first and last 10 rows
            continue
        if (h_mean / 255.0) < (1 - h_black_thresh):  # is it a border
            cv2.line(image, (0, y), (img_w, y), white, 1, 8)  # then make white

    # Remote left and right border
    for x, v_mean in enumerate(v_means):
        if 10 < x < (img_w - 10):  # Only consider the first and last 10 columns
            continue
        if (v_mean / 255.0) < (1 - v_black_thresh):  # is it a border
            cv2.line(image, (x, 0), (x, img_h), white, 1, 8)  # then make white

    return image


def tpl_sub(lhs, rhs):
    return lhs[0] - rhs[0], lhs[1] - rhs[1]


def tpl_add(lhs, rhs):
    return lhs[0] + rhs[0], lhs[1] + rhs[1]


def process_scores(scores):
    # Trying to find two vertically aligned points that are at a specific distance of one another.
    dy = 33
    h, w = scores.shape

    min_v = 9999999999
    loc = (0, 0)

    # x 78 ± 3
    # y 29 ± 5

    for x in range(w):
        for y in range(h - dy):
            if not (72 < x < 84 and 24 < y < 34):
                continue
            top_s = scores[y][x]
            btm_s = scores[y + dy][x]
            score = btm_s + top_s
            if score < min_v:
                min_v = score
                loc = (x, y)
    return loc


def write_over_hour_mark(image, loc):
    image = image.copy()
    dy = 32  # difference in height between top and bottom mark
    side = 9
    cv2.rectangle(image, (loc[0] + 0, loc[1] + 0), (loc[0] + side, loc[1] + side), 255, -1)  # -1 for fill
    cv2.rectangle(image, (loc[0] + 0, loc[1] + dy), (loc[0] + side, loc[1] + dy + side), 255, -1)  # -1 for fill
    return image


if __name__ == '__main__':
    image_wildcard, out_dir = get_opts(commands, usage)

    print("Stated cleaning with images from %s to %s" % (image_wildcard, out_dir))

    assert image_wildcard is not None, "Missing required image wildcard"
    assert out_dir is not None, "Missing required output directory"
    assert os.path.exists(out_dir), "Output directory does not exist."

    for file in tqdm(glob.glob(image_wildcard)):
        # Load as B&W
        img = as_grayscale(file)
        # Remove remaining border
        img = remove_border(img)
        # If blank skip (some might have dash through and we are keeping them)
        if is_blank(img):
            continue  # must be after remove border.
        # Give a score for the whole image where a hour mark dot could be.
        scores = cv2.matchTemplate(img, single_hour_mark, cv2.TM_SQDIFF)
        # Find best location for two dots aligned vertically and properly spaced.
        location = process_scores(scores)
        # Write over the hour mark (set pixels to white)
        img = write_over_hour_mark(img, location)

        cv2.imwrite("%s/%s" % (out_dir, os.path.basename(file)), img)

