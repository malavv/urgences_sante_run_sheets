import cv2

import glob
import os

import cv2
from tqdm import tqdm
import numpy as np

from helper import get_opts, as_grayscale

usage = "python models/max_mod_1.py --img <wildcard>"
commands = ["img", "out"]


def load_as_128x32(filename):
    # Since the original image is (176, 97)
    # so the closest size for this aspect ratio is (58, 32)
    # I want a 128x32 and paste the thing.
    gray = as_grayscale(filename)
    resized = cv2.resize(gray, (58, 32), interpolation=cv2.INTER_CUBIC)
    # Create blank and paste
    blank_image = np.zeros(shape=[32, 128], dtype=np.uint8)
    blank_image[0:32, 0:58] = resized
    return blank_image


if __name__ == '__main__':
    image_wildcard, out_dir = get_opts(commands, usage)

    print("Stated focusing with images from %s" % image_wildcard)

    assert image_wildcard is not None, "Missing required image wildcard"

    for file in tqdm(glob.glob(image_wildcard)):
        img = load_as_128x32(file)
