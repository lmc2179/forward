import numpy as np
from forward.backshift_difference import backshift

def autocovariance(data, lag):
    shifted_series = backshift(data, lag)
    acf = np.cov(shifted_series[lag:], data[lag:])[0][1]
    return acf

def autocorrelation(data, lag):
    return autocovariance(data, lag) / autocovariance(data, 0)