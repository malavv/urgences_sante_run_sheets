#!/usr/bin/env python

"""
Digit Segmenter
"""
import glob

from Helper import get_opts

usage = "python digit-segment/ --img <images wildcard> --out <output directory>"
commands = ["img", "out"]


if __name__ == '__main__':
    image_wildcard, out_dir = get_opts(commands, usage)

    images = glob.glob(image_wildcard)

    print("Starting Digit Segmenter with configuration: %s, %s, %s" % (image_wildcard, images, out_dir))
