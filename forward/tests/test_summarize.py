import unittest
import numpy as np
from forward.summarize import autocorrelation, autocovariance

class AutocorrelationTest(unittest.TestCase):
    def test_autocorrelation(self):
        data = np.array(np.random.normal(0, 100, 10000))
        self.assertEqual(autocorrelation(data, 0), 1)
        self.assertLess(autocorrelation(data, 1), autocorrelation(data, 0))

class AutocovarianceTest(unittest.TestCase):
    def test_autocovariance(self):
        data = np.array(np.random.normal(0, 100, 10000))
        self.assertAlmostEqual(autocovariance(data, 0), np.var(data, ddof=1))
        self.assertGreater(autocovariance(data, 0), autocovariance(data, 1))