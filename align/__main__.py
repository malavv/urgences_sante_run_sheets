#!/usr/bin/env python

import glob2
import os
import sys

from align.Align import Align
from align.Compositor import Compositor
from helper import get_opts

usage = "python digit-segment/ --ref <reference> --img <images> --out <directory>"
cmds = ["ref", "img", "out", "NUM_FEATURES", "GOOD_MATCH_PERCENT", "PERC_COMMON_THRESH", "COMPOSITE", "v", "DRY_RUN"]

if __name__ == '__main__':
    ref, img_pattern, out_dir, num_feat, good_match_perc, common_match_perc, composite, verbose, is_dry_run = get_opts(cmds, usage)

    assert os.path.exists(ref), "Reference image not found"
    assert os.path.exists(out_dir), "Output directory does not exist"

    images = glob2.glob(img_pattern)
    if len(images) == 0:
        print("Filename pattern matched no images")
        sys.exit()

    aligner = Align(ref, images, out_dir, Compositor.get_by_name(composite))

    # Run the alignment
    aligner.align(num_feat or 5000, good_match_perc or 0.4, common_match_perc or 0.4, verbose or False, is_dry_run or False)
