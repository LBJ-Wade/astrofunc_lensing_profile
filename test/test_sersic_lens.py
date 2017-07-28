__author__ = 'sibirrer'


from astrofunc.LensingProfiles.sersic import Sersic
from astrofunc.LightProfiles.sersic import Sersic as Sersic_light

import numpy as np
import pytest
import numpy.testing as npt


class TestSersic(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.sersic = Sersic()
        self.sersic_light = Sersic_light()


    def test_function(self):
        x = 1
        y = 2
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        values = self.sersic.function(x, y, n_sersic, r_eff, k_eff)
        npt.assert_almost_equal(values, 1.0272982586319199, decimal=10)

        x = np.array([0])
        y = np.array([0])
        values = self.sersic.function(x, y, n_sersic, r_eff, k_eff)
        npt.assert_almost_equal(values[0], 0., decimal=10)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.sersic.function(x, y, n_sersic, r_eff, k_eff)
        npt.assert_almost_equal(values[0], 1.0272982586319199, decimal=10)
        npt.assert_almost_equal(values[1], 1.3318743892966658, decimal=10)
        npt.assert_almost_equal(values[2], 1.584299393114988, decimal=10)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        f_x, f_y = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff)
        assert f_x[0] == 0.16556078301997193
        assert f_y[0] == 0.33112156603994386
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff)
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff)
        assert values[0][0] == 0.16556078301997193
        assert values[1][0] == 0.33112156603994386
        assert values[0][1] == 0.2772992378623737
        assert values[1][1] == 0.092433079287457892

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        f_xx, f_yy,f_xy = self.sersic.hessian(x, y, n_sersic, r_eff, k_eff)
        assert f_xx[0] == 0.15125869945524892
        npt.assert_almost_equal(f_yy[0], -0.086355912401166718, decimal=10)
        npt.assert_almost_equal(f_xy[0], -0.15840974123761042, decimal=10)
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.sersic.hessian(x, y, n_sersic, r_eff, k_eff)
        assert values[0][0] == 0.15125869945524892
        npt.assert_almost_equal(values[1][0], -0.086355912401166718, decimal=10)
        npt.assert_almost_equal(values[2][0], -0.15840974123761042, decimal=10)
        npt.assert_almost_equal(values[0][1], -0.071649513200062631, decimal=10)
        npt.assert_almost_equal(values[1][1], 0.094619015499099512, decimal=10)
        npt.assert_almost_equal(values[2][1], -0.062350698262185797, decimal=10)

    def test_all(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.sersic.all(x, y, n_sersic, r_eff, k_eff)
        npt.assert_almost_equal(f_[0], 1.0272982586319199, decimal=10)
        assert f_x[0] == 0.16556078301997193
        assert f_y[0] == 0.33112156603994386
        assert f_xx[0] == 0.15125869945524892
        npt.assert_almost_equal(f_yy[0], -0.086355912401166718, decimal=10)
        npt.assert_almost_equal(f_xy[0], -0.15840974123761042, decimal=10)

    def test_magnificaton(self):
        """

        :return:
        """
        r = 5.
        angle = 0.
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        n_sersic = 2.5
        r_eff = 2.5
        k_eff = 0.8
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.sersic.all(x, y, n_sersic, r_eff, k_eff)
        mag = (1. - f_xx) * (1. - f_yy) - f_xy**2

        p = self.sersic._p(x, y, n_sersic, r_eff, k_eff)
        q = self.sersic._q(x, y, n_sersic, r_eff, k_eff)
        A = (1-p)*(1+p+q)
        print(mag, A, 'mag, A')
        npt.assert_almost_equal(mag, A, decimal=10)

        d_alpha_dr = self.sersic.d_alpha_dr(x, y, n_sersic, r_eff, k_eff)
        assert -d_alpha_dr == p+q
        alpha = self.sersic.alpha_abs(x, y, n_sersic, r_eff, k_eff)
        npt.assert_almost_equal(alpha/r, p, decimal=10)
        mag_new = (1 - alpha / r) * (1 - d_alpha_dr)
        npt.assert_almost_equal(mag_new, A, decimal=10)
        print(mag)

        print(A, 'A')
        npt.assert_almost_equal(mag, A, decimal=10)

    def test_mag_sym(self):
        """

        :return:
        """
        r = 2.
        angle1 = 0.
        angle2 = 1.5
        x1 = r * np.cos(angle1)
        y1 = r * np.sin(angle1)

        x2 = r * np.cos(angle2)
        y2 = r * np.sin(angle2)
        n_sersic = 4.5
        r_eff = 2.5
        k_eff = 0.8
        f_1, f_x1, f_y1, f_xx1, f_yy1, f_xy1 = self.sersic.all(x1, y1, n_sersic, r_eff, k_eff)
        f_2, f_x2, f_y2, f_xx2, f_yy2, f_xy2 = self.sersic.all(x2, y2, n_sersic, r_eff, k_eff)
        kappa_1 = (f_xx1 + f_yy1) / 2
        kappa_2 = (f_xx2 + f_yy2) / 2
        npt.assert_almost_equal(kappa_1, kappa_2, decimal=10)
        A_1 = (1 - f_xx1) * (1 - f_yy1) - f_xy1**2
        A_2 = (1 - f_xx2) * (1 - f_yy2) - f_xy2 ** 2
        npt.assert_almost_equal(A_1, A_2, decimal=10)

    def test_convergernce(self):
        """
        test the convergence and compares it with the original Sersic profile
        :return:
        """
        x = np.array([0, 0, 0, 0, 0])
        y = np.array([0.5, 1, 1.5, 2, 2.5])
        n_sersic = 4.5
        r_eff = 2.5
        k_eff = 0.2
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.sersic.all(x, y, n_sersic, r_eff, k_eff)
        kappa = (f_xx + f_yy) / 2.
        assert kappa[0] > 0
        flux = self.sersic_light.function(x, y, I0_sersic=1., R_sersic=r_eff, n_sersic=n_sersic)
        flux /= flux[0]
        kappa /= kappa[0]
        npt.assert_almost_equal(flux[1], kappa[1], decimal=10)

if __name__ == '__main__':
    pytest.main()