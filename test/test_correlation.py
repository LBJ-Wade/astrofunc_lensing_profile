__author__ = 'sibirrer'


from astrofunc.correlation import Correlation

import numpy as np
import pytest
#from lenstronomy.unit_manager import UnitManager

class TestCorrelation(object):

    def setup(self):
        self.correlation = Correlation()

    def test_corr1D(self):
        residuals = np.ones((10,10))
        residuals[5,5] = 100
        psd1D, psd2D = self.correlation.correlation_2D(residuals)
        assert psd1D[0] == 99


if __name__ == '__main__':
    pytest.main()