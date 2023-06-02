import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import datetime
from operator import itemgetter
from utils.es.fetch import fetchDocsByPeriod

from utils.q import calc_q_kw
import sys
import os

import numpy as np
from itertools import chain


if __name__ == "__main__":
    def full_cross_correlation(a, b):
        len_a = len(a)
        len_b = len(b)
        out_len = len_a + len_b - 1
        out = np.zeros(out_len)

        for i in range(out_len):
            for j in range(max(0, i - len_b + 1), min(i, len_a - 1) + 1):
                out[i] += a[j] * b[i - j]
                print(i, j)

        return out + full_cross_correlation(b[::-1], a)
    
    full_cross_correlation(np.array([0, 1, 2]), np.array([3, 4, 5]))
