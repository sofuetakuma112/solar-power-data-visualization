import numpy as np


def min_max(values):
    values_nparray = np.array(values)
    min_value = np.amin(values_nparray)
    max_value = np.amax(values_nparray)
    return (values_nparray - min_value) / (max_value - min_value)
