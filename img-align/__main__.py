#!/usr/bin/env python

"""
proc.py: Image Perspective Alignment

From the article on: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

usage:
  proc --ref "../imgs/21543780.png" --img "../imgs/*.png" --out "../aligned/"

"""
import glob
import os
import sys
import getopt

from proc.Align import Align
from proc.Compositor import Compositor


def usage():
    """How to call this script"""
    print("proc.py --ref <ref_image> --img <images> --out <output directory>")


def get_opt():
    """Process Arguments to this script"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hr:i:o:", ["help", "ref=", "img=", "out="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    reference_filename = None
    images_filename = None
    out_directory = None
    # Process options
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-r", "--ref"):
            reference_filename = a
        elif o in ("-i", "--img"):
            images_filename = a
        elif o in ("-o", "--out"):
            out_directory = a
        else:
            assert False, "unhandled option"
    # Missing Parameters
    if reference_filename is None or images_filename is None or out_directory is None:
        usage()
        sys.exit()

    return reference_filename, images_filename, out_directory


if __name__ == '__main__':
    ref_filename, imgs_filename, out_dir = get_opt()

    assert os.path.exists(ref_filename), "Reference image not found"
    assert os.path.exists(out_dir), "Output directory does not exist"

    imgs = glob.glob(imgs_filename)
    assert len(imgs) > 0, "Must have at least one image to convert."

    compositor = Compositor.get_additive()

    aligner = Align(ref_filename, imgs, out_dir, compositor)

    aligned_dic = aligner.align()

