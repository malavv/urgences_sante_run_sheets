import getopt
import sys


def get_opts(avail_commands, fun_usage):

    first_letters = [cmd[0] for cmd in avail_commands]
    shortcuts = "h" + "".join([l + ":" for l in first_letters])
    full = ["help"] + [cmd + "=" for cmd in avail_commands]

    try:
        opts, args = getopt.getopt(sys.argv[1:], shortcuts, full)
    except getopt.GetoptError as err:
        print(err)  # will print something like "option -a not recognized"
        print(fun_usage)
        sys.exit(2)

    configurations = [None for c in avail_commands]

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


def get_overlap(lhs, rhs) -> float:
    """
    Get percentage of overlap between rectangle lhs, and rhs
    :param lhs: 2D rectangle [x, y, w, h]
    :param rhs: 2D rectangle [x, y, w, h]
    :return: Percent overlap
    """
    x1, y1, w1, h1 = lhs
    x2, y2, w2, h2 = rhs

    # Some stackoverflow code to compute overlap of rectangle A and B. Seem to come from VB, ugly but meh.
    SA = w1 * h1
    SB = w2 * h2
    SI = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    SU = SA + SB - SI
    perc_overlap = SI / float(SU)

    return perc_overlap
