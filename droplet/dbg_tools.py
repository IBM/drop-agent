import pdb
import sys
import traceback


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    pdb.pm()  # post-mortem debugger

def pm_breakpoint():
    sys.excepthook = debughook
