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
show_debug_vis = False


def as_grayscale(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def is_mostly_blank(img):
    return np.mean(img) > 240


if __name__ == '__main__':
    image_wildcard, out_dir = get_opts(commands, usage)

    images = glob.glob(image_wildcard)

    segmenter = DigitSegmenter("./digit-segment/hour_mask.png", show_debug_vis)

    for img in tqdm(images):
        base = os.path.basename(img)
        gray = as_grayscale(img)

        if is_mostly_blank(gray):
            continue

        hstack = segmenter.segment(gray)
        cv2.imwrite("%s/%s" % (out_dir, base), hstack)
