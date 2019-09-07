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
