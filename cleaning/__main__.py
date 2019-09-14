#!/usr/bin/env python

"""
Digit Segmenter
"""
import glob
from tqdm import tqdm
import numpy as np

from helper import get_opts, as_grayscale

usage = "python cleaning/ --img <wildcard> --out <dir>"
commands = ["img", "out"]

blank_thresh = 237  # gotten from an histogram of ~10000 cropped images


def is_blank(image):
    return np.mean(image) > blank_thresh


if __name__ == '__main__':
    image_wildcard, out_dir = get_opts(commands, usage)

    assert image_wildcard is not None, "Missing required image wildcard"
    assert out_dir is not None, "Missing required output directory"

    print("Stated cleaning with images from %s to %s" % (image_wildcard, out_dir))

    res = []
    for file in tqdm(glob.glob(image_wildcard)):
        img = as_grayscale(file)

        if is_blank(img):
            continue  # is blank

    t = np.array(res)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(t, bins=150, density=True)
    plt.show()
    print("test")

