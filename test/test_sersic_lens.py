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
        assert f_x[0] == 0.027593463836661988
        assert f_y[0] == 0.055186927673323977
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff)
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff)
        assert values[0][0] == 0.027593463836661988
        assert values[1][0] == 0.055186927673323977
        assert values[0][1] == 0.046216539643728946
        assert values[1][1] == 0.015405513214576316

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        f_xx, f_yy,f_xy = self.sersic.hessian(x, y, n_sersic, r_eff, k_eff)
        assert f_xx[0] == -0.095077789220926703
        assert f_yy[0] == 0.15998057627500889
        assert f_xy[0] == 0.17003891033062374
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.sersic.hessian(x, y, n_sersic, r_eff, k_eff)
        assert values[0][0] == -0.095077789220926703
        assert values[1][0] == 0.15998057627500889
        assert values[2][0] == 0.17003891033062374
        assert values[0][1] == 0.091424424786632347
        assert values[1][1] == -0.068454922487595479
        assert values[2][1] == 0.059954755227835436

    def test_all(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.sersic.all(x, y, n_sersic, r_eff, k_eff)
        npt.assert_almost_equal(f_[0], 1.0272982586319199, decimal=10)
        assert f_x[0] == 0.027593463836661988
        assert f_y[0] == 0.055186927673323977
        assert f_xx[0] == -0.095077789220926703
        assert f_yy[0] == 0.15998057627500889
        assert f_xy[0] == 0.17003891033062374

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
        flux = self.sersic_light.function(x, y, I0_sersic=1., R_sersic=r_eff, n_sersic=n_sersic)
        flux /= flux[0]
        kappa /= kappa[0]
        npt.assert_almost_equal(flux[1], kappa[1], decimal=10)


if __name__ == '__main__':
    pytest.main()