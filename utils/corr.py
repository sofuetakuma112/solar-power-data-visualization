import numpy as np


def calc_delay(a, b):
    # aを固定して、bを左から右へスライドさせていく
    corr = np.correlate(a, b, "full")
    return [corr, corr.argmax() - (len(b) - 1)]
