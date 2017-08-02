__author__ = 'sibirrer'

import pytest
import numpy.testing as npt

from astrofunc.numeric_lens_differentials import NumericLens


class TestNumerics(object):
    """
    tests the second derivatives of various lens models
    """
    def setup(self):
        pass

    def assert_differentials(self, Model, kwargs):
        lensModel = Model()
        lensModelNum = NumericLens(lensModel, diff=0.00000001)
        x, y = 1., 2.
        f_x, f_y = lensModel.derivatives(x, y, **kwargs)
        f_xx, f_yy, f_xy = lensModel.hessian(x, y, **kwargs)
        f_x_num, f_y_num = lensModelNum.derivatives(x, y, kwargs)
        f_xx_num, f_yy_num, f_xy_num = lensModelNum.hessian(x, y, kwargs)
        print(f_xx_num, f_xx)
        print(f_yy_num, f_yy)
        print(f_xy_num, f_xy)
        print((f_xx - f_yy)**2/4 + f_xy**2, (f_xx_num - f_yy_num)**2/4 + f_xy_num**2)
        npt.assert_almost_equal(f_x, f_x_num, decimal=5)
        npt.assert_almost_equal(f_y, f_y_num, decimal=5)
        npt.assert_almost_equal(f_xx, f_xx_num, decimal=3)
        npt.assert_almost_equal(f_yy, f_yy_num, decimal=3)
        npt.assert_almost_equal(f_xy, f_xy_num, decimal=3)

    def test_gaussian(self):
        kwargs = {'amp': 1. / 4., 'sigma_x': 2., 'sigma_y': 2., 'center_x': 0., 'center_y': 0.}
        from astrofunc.LensingProfiles.gaussian import Gaussian as Model
        self.assert_differentials(Model, kwargs)

    # TODO fix test
    def test_external_shear(self):
        kwargs = {'e1': 0.1, 'e2': -0.1}
        from astrofunc.LensingProfiles.external_shear import ExternalShear as Model
        self.assert_differentials(Model, kwargs)

    def test_sis(self):
        kwargs = {'theta_E': 0.5}
        from astrofunc.LensingProfiles.sis import SIS as Model
        self.assert_differentials(Model, kwargs)

    def test_flexion(self):
        kwargs = {'g1': 0.01, 'g2': -0.01, 'g3': 0.001, 'g4': 0}
        from astrofunc.LensingProfiles.flexion import Flexion as Model
        self.assert_differentials(Model, kwargs)

    def test_nfw(self):
        kwargs = {'theta_Rs': 1., 'Rs': 5.}
        from astrofunc.LensingProfiles.nfw import NFW as Model
        self.assert_differentials(Model, kwargs)

    def test_nfw_ellipse(self):
        kwargs = {'theta_Rs': 1., 'Rs': 5., 'q': 0.9, 'phi_G': 0.}
        from astrofunc.LensingProfiles.nfw_ellipse import NFW_ELLIPSE as Model
        self.assert_differentials(Model, kwargs)

    def test_point_mass(self):
        kwargs = {'theta_E': 1.}
        from astrofunc.LensingProfiles.point_mass import PointMass as Model
        self.assert_differentials(Model, kwargs)

    def test_sersic(self):
        kwargs = {'n_sersic': .5, 'r_eff': 1.5, 'k_eff': 0.3}
        from astrofunc.LensingProfiles.sersic import Sersic as Model
        self.assert_differentials(Model, kwargs)

    def test_sersic_ellipse(self):
        kwargs = {'n_sersic': 2., 'r_eff': 0.5, 'k_eff': 0.3, 'q': 0.9, 'phi_G': 1.}
        from astrofunc.LensingProfiles.sersic_ellipse import SersicEllipse as Model
        self.assert_differentials(Model, kwargs)

    def test_shapelets_pot_2(self):
        kwargs = {'coeffs': [0, 1, 2, 3, 4, 5], 'beta': 0.3}
        from astrofunc.LensingProfiles.shapelet_pot_2 import CartShapelets as Model
        self.assert_differentials(Model, kwargs)

    def test_sis_truncate(self):
        kwargs = {'theta_E': 0.5, 'r_trunc': 2.}
        from astrofunc.LensingProfiles.sis_truncate import SIS_truncate as Model
        self.assert_differentials(Model, kwargs)

    def test_spemd(self):
        kwargs = {'theta_E': 0.5, 'gamma': 1.9, 'q': 0.8, 'phi_G': 1.}
        from astrofunc.LensingProfiles.spemd import SPEMD as Model
        self.assert_differentials(Model, kwargs)

    def test_spep(self):
        kwargs = {'theta_E': 0.5, 'gamma': 1.9, 'q': 0.8, 'phi_G': 1.}
        from astrofunc.LensingProfiles.spep import SPEP as Model
        self.assert_differentials(Model, kwargs)

    def test_spp(self):
        kwargs = {'theta_E': 0.5, 'gamma': 1.9}
        from astrofunc.LensingProfiles.spp import SPP as Model
        self.assert_differentials(Model, kwargs)

    def test_PJaffe(self):
        kwargs = {'sigma0': 1., 'a': 0.2, 's': 2.}
        from astrofunc.LensingProfiles.p_jaffe import PJaffe as Model
        self.assert_differentials(Model, kwargs)

    def test_Hernquist(self):
        kwargs = {'sigma0': 1., 'Rs': 1.5}
        from astrofunc.LensingProfiles.hernquist import Hernquist as Model
        self.assert_differentials(Model, kwargs)

if __name__ == '__main__':
    pytest.main("-k TestLensModel")