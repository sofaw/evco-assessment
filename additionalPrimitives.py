from functools import partial

def if_then_else(condition, out1, out2):
    out1() if condition() else out2()


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn,out1,out2)


def prog3(out1, out2, out3):
    return partial(progn,out1,out2,out3)
