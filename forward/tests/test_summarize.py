import unittest
import numpy as np
from forward.summarize import autocorrelation, autocovariance

class AutocorrelationTest(unittest.TestCase):
    def test_autocorrelation_iid_noise(self):
        data = np.array(np.random.normal(0, 100, 10000))
        self.assertEqual(autocorrelation(data, 0), 1)
        self.assertLess(autocorrelation(data, 1), autocorrelation(data, 0))

    def test_autocorrelation_range(self):
        data = np.array(range(10000))
        self.assertEqual(autocorrelation(data, 0), 1)
        self.assertAlmostEqual(autocorrelation(data, 1), 1.0, delta=0.002)
        self.assertAlmostEqual(autocorrelation(data, 2), 1.0, delta=0.002)
        self.assertAlmostEqual(autocorrelation(data, 10), 1.0, delta=0.002)
        self.assertLess(autocorrelation(data, 1), autocorrelation(data, 0))

class AutocovarianceTest(unittest.TestCase):
    def test_autocovariance(self):
        data = np.array(np.random.normal(0, 1, 10000))
        self.assertAlmostEqual(autocovariance(data, 0), np.var(data, ddof=1))
        self.assertGreater(autocovariance(data, 0), autocovariance(data, 1))
        averages = np.average([autocovariance(data, i) for i in range(1000)])
        self.assertAlmostEqual(averages, 0, delta=0.001)