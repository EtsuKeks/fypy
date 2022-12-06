import unittest

from tests.pricing.fourier.suite import test_suite as fourier_suite
from tests.fit.suite import test_suite as fit_suite
from tests.volatility.suite import test_suite as vol_suite

##############################################


def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(fourier_suite())
    suite.addTests(fit_suite())
    suite.addTests(vol_suite())
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(test_suite())
