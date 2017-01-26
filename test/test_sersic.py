__author__ = 'sibirrer'


from astrofunc.LightProfiles.sersic import Sersic, Sersic_elliptic, DoubleSersic, CoreSersic

import numpy as np
import pytest
import numpy.testing as npt

class TestSersic(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.sersic = Sersic()
        self.sersic_elliptic = Sersic_elliptic()
        self.double_sersic = DoubleSersic()
        self.core_sersic = CoreSersic()

    def test_sersic(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        center_x = 0
        center_y = 0
        values = self.sersic.function(x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.12658651833626802, decimal=6)
        x = np.array([0])
        y = np.array([0])
        values = self.sersic.function( x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y)
        npt.assert_almost_equal(values[0],  5.3233350710888789, decimal=6)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.sersic.function( x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.12658651833626802, decimal=6)
        npt.assert_almost_equal(values[1], 0.026902273598180083, decimal=6)
        npt.assert_almost_equal(values[2], 0.0053957432862338055, decimal=6)

    def test_sersic_elliptic(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        phi_G = 1
        q = 0.9
        center_x = 0
        center_y = 0
        values = self.sersic_elliptic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.12595366113005077, decimal=6)
        x = np.array([0])
        y = np.array([0])
        values = self.sersic_elliptic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 5.3233350710888789, decimal=6)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.sersic_elliptic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.11308277793465012, decimal=6)
        npt.assert_almost_equal(values[1], 0.021188620675507107, decimal=6)
        npt.assert_almost_equal(values[2], 0.0037276744362724477, decimal=6)

    def test_core_sersic(self):
        x = np.array([1])
        y = np.array([2])
        I0 = 1
        Rb = 1
        Re = 2
        gamma = 3
        n = 1
        phi_G = 1
        q = 0.9
        center_x = 0
        center_y = 0
        values = self.core_sersic.function(x, y, I0, Rb, Re, n, gamma, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.84489101, decimal=8)
        x = np.array([0])
        y = np.array([0])
        values = self.core_sersic.function(x, y, I0, Rb, Re, n, gamma, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 2307237.2, decimal=0)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.core_sersic.function(x, y, I0, Rb, Re, n, gamma, phi_G, q, center_x, center_y)
        npt.assert_almost_equal(values[0], 0.79749529635325933, decimal=6)
        npt.assert_almost_equal(values[1], 0.33653478121594838, decimal=6)
        npt.assert_almost_equal(values[2], 0.14050402887681532, decimal=6)

    def test_double_sersic(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        phi_G = 1
        q = 0.9
        I0_2 = 0.1
        R_2 = 2
        n_2 = 2
        center_x_2 = 1
        center_y_2 = 2
        center_x = 0
        center_y = 0
        values = self.double_sersic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y, I0_2, R_2, n_2, center_x_2, center_y_2)
        assert values[0] == 4.0562500804662704
        x = np.array([0])
        y = np.array([0])
        values = self.double_sersic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y, I0_2, R_2, n_2, center_x_2, center_y_2)
        assert values[0] == 5.40434230864048

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.double_sersic.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y, I0_2, R_2, n_2, center_x_2, center_y_2)
        npt.assert_almost_equal(values[0], 0.29242342710494995, decimal=8)
        npt.assert_almost_equal(values[1], 0.10219623707234859, decimal=8)
        npt.assert_almost_equal(values[2], 0.042591148350819084, decimal=8)

if __name__ == '__main__':
    pytest.main()