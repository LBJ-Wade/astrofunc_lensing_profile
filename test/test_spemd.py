__author__ = 'sibirrer'


from astrofunc.LensingProfiles.spemd import SPEMD

import numpy as np
import pytest
import numpy.testing as npt

class TestSPEP(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.SPEMD = SPEMD()

    def test_function(self):
        phi_E = 1.
        gamma = 1.9
        q = 0.9
        phi_G = 1.
        x = np.array([1])
        y = np.array([2])
        values = self.SPEMD.function(x, y, phi_E, gamma, q, phi_G)
        assert values == 1.911701618701235

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.SPEMD.function(x, y, phi_E, gamma, q, phi_G)
        npt.assert_almost_equal(values[0], 1.93472805144388, decimal=7)
        npt.assert_almost_equal(values[1], 2.9389573854207987, decimal=7)
        npt.assert_almost_equal(values[2], 4.0207564579911459, decimal=7)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 0.9
        phi_G = 1.
        f_x, f_y = self.SPEMD.derivatives(x, y, phi_E, gamma, q, phi_G)
        npt.assert_almost_equal(f_x[0], 0.45213064585508395, decimal=7)
        npt.assert_almost_equal(f_y[0], 0.92402362693408957, decimal=7)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.SPEMD.derivatives(x, y, phi_E, gamma, q, phi_G)
        npt.assert_almost_equal(values[0][0], 0.45213064585508395, decimal=7)
        npt.assert_almost_equal(values[1][0], 0.92402362693408957, decimal=7)
        npt.assert_almost_equal(values[0][1], 1.0502477692714798, decimal=7)
        npt.assert_almost_equal(values[1][1], 0.30672878119865871, decimal=7)

        x = 1
        y = 2
        phi_E = 1.
        gamma = 1.9
        q = 0.9
        phi_G = 1.
        f_x, f_y = self.SPEMD.derivatives(x, y, phi_E, gamma, q, phi_G)
        npt.assert_almost_equal(f_x, 0.45213064585508395, decimal=7)
        npt.assert_almost_equal(f_y, 0.92402362693408957, decimal=7)
        x = 0
        y = 0
        f_x, f_y = self.SPEMD.derivatives(x, y, phi_E, gamma, q, phi_G)
        assert f_x == 0
        assert f_y == 0

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 0.9
        phi_G = 1.
        f_xx, f_yy,f_xy = self.SPEMD.hessian(x, y, phi_E, gamma, q, phi_G)
        npt.assert_almost_equal(f_xx, 0.40902479080999932, decimal=7)
        npt.assert_almost_equal(f_yy, 0.1488504387799334, decimal=7)
        npt.assert_almost_equal(f_xy, -0.17413533543756601, decimal=7)
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.SPEMD.hessian(x, y, phi_E, gamma, q, phi_G)
        npt.assert_almost_equal(values[0][0], 0.40902479080999932, decimal=7)
        npt.assert_almost_equal(values[1][0], 0.1488504387799334, decimal=7)
        npt.assert_almost_equal(values[2][0], -0.17413533543756601, decimal=7)
        npt.assert_almost_equal(values[0][1], 0.074088256439004602, decimal=7)
        npt.assert_almost_equal(values[1][1], 0.3190504949856876, decimal=7)
        npt.assert_almost_equal(values[2][1], -0.09352224663857206, decimal=7)

    def test_all(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 0.9
        phi_G = 1.
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.SPEMD.all(x, y, phi_E, gamma, q, phi_G)
        npt.assert_almost_equal(f_[0], 2.1571109390326626, decimal=7)
        npt.assert_almost_equal(f_x[0], 0.45213064585508395, decimal=7)
        npt.assert_almost_equal(f_y[0], 0.92402362693408957, decimal=7)
        npt.assert_almost_equal(f_xx[0], 0.40902479080999932, decimal=7)
        npt.assert_almost_equal(f_yy[0], 0.1488504387799334, decimal=7)
        npt.assert_almost_equal(f_xy[0], -0.17413533543756601, decimal=7)

if __name__ == '__main__':
   pytest.main()