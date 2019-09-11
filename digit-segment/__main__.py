#!/usr/bin/env python

"""
Digit Segmenter
"""
import glob
import cv2
import os
from tqdm import tqdm
import numpy as np


from segment.DigitSegmenter import DigitSegmenter
from segment.Helper import get_opts

usage = "python digit-segment/ --img <images wildcard> --out <output directory>"
commands = ["img", "out"]
# Show a image visualization for each step of the process
show_debug_vis = False
# Means images with a 1% area over will be considered similar, and duplicate removed
overlap_perc_for_similar_threshold = 0.1
# A px padding will be created around the square digit before resizing.
digit_padding_in_px = 5


def as_grayscale(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def is_mostly_blank(img):
    return np.mean(img) > 240


def is_digit_shaped(rect):
    x, y, w, h = rect
    return 3 < h and 3 < w < 50  # Complete heuristic


if __name__ == '__main__':
    image_wildcard, out_dir = get_opts(commands, usage)

    images = glob.glob(image_wildcard)

    segmenter = DigitSegmenter("./digit-segment/hour_mask.png", show_debug_vis, overlap_perc_for_similar_threshold,
                               digit_padding_in_px, is_digit_shaped)

    for img in tqdm(images):
        base = os.path.basename(img)
        gray = as_grayscale(img)

        if is_mostly_blank(gray):
            continue

        hstack = segmenter.segment(gray)
        cv2.imwrite("%s/%s" % (out_dir, base), hstack)
