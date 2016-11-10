import numpy as np


def backshift(data, positions=1):
    backshifted_data = np.nan * np.empty(data.shape)
    backshifted_data[positions:] = data[:-positions]
    return backshifted_data

def lag_difference(data):
    return data - backshift(data)