#!/usr/bin/env python

"""
Batch apply some cropping configuration, Saving cropped images separately.

usage:
  python batch-crop/ --cfg ./batch-crop/crops.yaml --img ../data/us/forms/*.png --out ../data/us/cropped-forms
"""

import getopt
import glob
import os
import sys
import yaml

from Cropper import Cropper


def usage():
    """How to call this script"""
    print("python batch-crop/ --cfg <config file> --img <images wildcard> --out <output directory>")


def get_opt():
    """Process Arguments to this script"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hc:i:o:", ["help", "cfg=", "img=", "out="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cfg_filename = None
    images_filename = None
    out_directory = None
    # Process options
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-c", "--cfg"):
            cfg_filename = a
        elif o in ("-i", "--img"):
            images_filename = a
        elif o in ("-o", "--out"):
            out_directory = a
        else:
            assert False, "unhandled option"
    # Missing Parameters
    if cfg_filename is None or images_filename is None or out_directory is None:
        usage()
        sys.exit()

    return cfg_filename, images_filename, out_directory


def load_config(cfg_filename):
    with open(cfg_filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            sys.exit(exc)


if __name__ == '__main__':
    config_filename, imgs_filename, out_dir = get_opt()

    assert os.path.exists(config_filename), "Configuration File image not found"
    assert os.path.exists(out_dir), "Output directory does not exist"

    images = glob.glob(imgs_filename)
    assert len(images) > 0, "Must have at least one image to convert."

    cropper = Cropper(images, out_dir)
    config = load_config(config_filename)

    print("Starting cropping of %s files" % len(images))
    cropper.crop(config)
