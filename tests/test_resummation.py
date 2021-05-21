import unittest
import numpy as np
from series.resummation import resum_series
# from series.resummation import sum_series
from series.resummation import EulerTransform

class TestResumSeries(unittest.TestCase):

    def test_A(self):
        a = (-1.) ** np.arange(10) # a = 1 / (X + 1)
        phi = EulerTransform(-1.)
        a_resum = resum_series(a, phi)
        res = np.zeros(10)
        res[:2] = [1., -1.]
        np.testing.assert_allclose(a_resum, res)


if __name__ == '__main__':
    unittest.main()
