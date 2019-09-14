import getopt
import sys
from typing import List, Tuple, Any

import cv2

# Declare a rect type annotation using a tuple of ints of [x, y, w, h]
Rect = Tuple[int, int, int, int]


def get_opts(avail_commands: List[str], fun_usage: str) -> List[object]:
    """
    Basic CLI Argument Processing

    Example usage: get_opts(["img", "out"], "python segment/ --img <img> --out <dir>")

    :param avail_commands: Array of long form commands
    :param fun_usage: Usage line for the user
    :return: Tuple processed arguments in order of avail_commands
    """
    first_letters = [cmd[0] for cmd in avail_commands]
    # Command shortcut configuration
    shortcuts = "h" + "".join([l + ":" for l in first_letters])
    # Full command configuration
    full = ["help"] + [cmd + "=" for cmd in avail_commands]

    try:
        opts, args = getopt.getopt(sys.argv[1:], shortcuts, full)
    except getopt.GetoptError as err:
        print(err)  # will print something like "option -a not recognized"
        print(fun_usage)
        sys.exit(2)

    configurations = [None for c in avail_commands]

    # asked for help
    for o, a in opts:
        if o in ("-h", "--help"):
            print(fun_usage)
            sys.exit()

    # dynamic commands
    for idx, cmd in enumerate(avail_commands):
        for o, a in opts:
            if o in ("-" + first_letters[0], "--" + cmd):
                configurations[idx] = a

    return configurations


def as_grayscale(filename: str) -> Any:
    """
    Loads grayscale image from filename.
    :param filename: Image full filename
    :return: OpenCV Loaded Image with 1 channel.
    """
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def get_overlap(lhs: Rect, rhs: Rect) -> float:
    """
    Get percentage of overlap between rectangle lhs, and rhs
    :param lhs: 2D rectangle [x, y, w, h]
    :param rhs: 2D rectangle [x, y, w, h]
    :return: Percent overlap
    """
    x1, y1, w1, h1 = lhs
    x2, y2, w2, h2 = rhs

    lhs_area = w1 * h1
    rhs_area = w2 * h2

    intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    union_area = lhs_area + rhs_area - intersection_area

    return intersection_area / float(union_area)
