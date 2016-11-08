import numpy as np


def backshift(data):
    backshifted_data = np.nan * np.empty(data.shape)
    backshifted_data[1:] = data[:-1]
    return backshifted_data

def lag_difference(data):
    return data - backshift(data)