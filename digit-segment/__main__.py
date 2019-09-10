#!/usr/bin/env python

"""
Digit Segmenter
"""
import glob
import cv2
import os

from DigitSegmenter import DigitSegmenter
from Helper import get_opts

usage = "python digit-segment/ --img <images wildcard> --out <output directory>"
commands = ["img", "out"]
show_debug_vis = False


def as_grayscale(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


if __name__ == '__main__':
    image_wildcard, out_dir = get_opts(commands, usage)

    images = glob.glob(image_wildcard)

    print("Starting Digit Segmenter with configuration: %s, %s, %s" % (image_wildcard, images, out_dir))

    segmenter = DigitSegmenter("hour_mask.png", show_debug_vis)

    for img in images:
        base = os.path.basename(img)
        hstack = segmenter.segment(as_grayscale(img))
        cv2.imwrite("%s/%s" % (out_dir, base), hstack)
