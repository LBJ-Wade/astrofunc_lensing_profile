__author__ = 'sibirrer'

from astrofunc.LensingProfiles.gaussian import Gaussian

import numpy as np
import pytest

class TestGaussian(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.Gaussian = Gaussian()


    def test_function(self):
        x = 1
        y = 2
        amp = 1.*2*np.pi
        center_x = 1.
        center_y = 1.
        sigma_x = 1.
        sigma_y = 1.
        values = self.Gaussian.function( x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values == np.exp(-1./2)
        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.Gaussian.function( x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values[0] == np.exp(-1./2)
        assert values[1] == np.exp(-2.**2/2)
        assert values[2] == np.exp(-3.**2/2)

    def test_derivatives(self):
        x = 1
        y = 2
        amp = 1.*2*np.pi
        center_x = 1.
        center_y = 1.
        sigma_x = 1.
        sigma_y = 1.
        values = self.Gaussian.derivatives( x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values[0] == 0.
        assert values[1] == -np.exp(-1./2)
        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.Gaussian.derivatives( x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values[0][0] == -np.exp(-1./2)
        assert values[1][0] == 0.
        assert values[0][1] == -2*np.exp(-2.**2/2)
        assert values[1][1] == 0.

    def test_hessian(self):
        x = 1
        y = 2
        amp = 1.*2*np.pi
        center_x = 1.
        center_y = 1.
        sigma_x = 1.
        sigma_y = 1.
        values = self.Gaussian.hessian( x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values[0] == -np.exp(-1./2)
        assert values[1] == 0.
        assert values[2] == 0.
        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.Gaussian.hessian( x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values[0][0] == 0.
        assert values[1][0] == -np.exp(-1./2)
        assert values[2][0] == 0.
        assert values[0][1] == 0.40600584970983811
        assert values[1][1] == -0.1353352832366127
        assert values[2][1] == 0.

    def test_all(self):
        x = 1
        y = 2
        amp = 1.*2*np.pi
        center_x = 1.
        center_y = 1.
        sigma_x = 1.
        sigma_y = 1.
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.Gaussian.all( x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert f_ == np.exp(-1./2)
        assert f_x == 0.
        assert f_y == -np.exp(-1./2)
        assert f_xx == -np.exp(-1./2)
        assert f_yy == 0.
        assert f_xy == 0.

if __name__ == '__main__':
    pytest.main()