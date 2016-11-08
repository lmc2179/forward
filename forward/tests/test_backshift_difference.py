import unittest
import numpy as np
from forward import backshift_difference

class BackshiftTest(unittest.TestCase):
    def test_single_backshift(self):
        A = np.array([2, 4, 6])
        A_shifted = backshift_difference.backshift(A)
        A_shifted = [x if not np.isnan(x) else None for x in A_shifted]
        self.assertEqual(list(A_shifted), [None, 2, 4])

class DifferenceTest(unittest.TestCase):
    def test_single_difference(self):
        A = np.array([2, 4, 6])
        A_shifted = backshift_difference.lag_difference(A)
        A_shifted = [x if not np.isnan(x) else None for x in A_shifted]
        self.assertEqual(list(A_shifted), [None, 2, 2])