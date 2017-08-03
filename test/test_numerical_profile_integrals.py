__author__ = 'sibirrer'

import pytest
import numpy.testing as npt

from astrofunc.numerical_profile_integrals import ProfileIntegrals


class TestNumerics(object):
    """
    tests the second derivatives of various lens models
    """
    def setup(self):
        pass

    def assert_integrals(self, Model, kwargs):
        lensModel = Model()
        int_profile = ProfileIntegrals(lensModel)
        r = 2.

        density2d_num = int_profile.density_2d(r, kwargs)
        density2d = lensModel.density_2d(r, 0, **kwargs)
        npt.assert_almost_equal(density2d, density2d_num, decimal=5)

        mass_2d_num = int_profile.mass_enclosed_2d(r, kwargs)
        mass_2d = lensModel.mass_2d(r, **kwargs)
        npt.assert_almost_equal(mass_2d, mass_2d_num, decimal=5)

        mass_3d_num = int_profile.mass_enclosed_3d(r, kwargs)
        mass_3d = lensModel.mass_3d(r, **kwargs)
        npt.assert_almost_equal(mass_3d, mass_3d_num, decimal=5)

    def test_PJaffe(self):
        kwargs = {'rho0': 1., 'Ra': 0.2, 'Rs': 2.}
        from astrofunc.LensingProfiles.p_jaffe import PJaffe as Model
        self.assert_integrals(Model, kwargs)

    """



    def test_gaussian(self):
        kwargs = {'amp': 1. / 4., 'sigma_x': 2., 'sigma_y': 2., 'center_x': 0., 'center_y': 0.}
        from astrofunc.LensingProfiles.gaussian import Gaussian as Model
        self.assert_integrals(Model, kwargs)

    def test_sis(self):
        kwargs = {'theta_E': 0.5}
        from astrofunc.LensingProfiles.sis import SIS as Model
        self.assert_integrals(Model, kwargs)

    def test_nfw(self):
        kwargs = {'theta_Rs': 1., 'Rs': 5.}
        from astrofunc.LensingProfiles.nfw import NFW as Model
        self.assert_integrals(Model, kwargs)

    def test_sersic(self):
        kwargs = {'n_sersic': .5, 'r_eff': 1.5, 'k_eff': 0.3}
        from astrofunc.LensingProfiles.sersic import Sersic as Model
        self.assert_integrals(Model, kwargs)

    def test_spep(self):
        kwargs = {'theta_E': 0.5, 'gamma': 1.9, 'q': 0.8, 'phi_G': 1.}
        from astrofunc.LensingProfiles.spep import SPEP as Model
        self.assert_integrals(Model, kwargs)

    def test_spp(self):
        kwargs = {'theta_E': 0.5, 'gamma': 1.9}
        from astrofunc.LensingProfiles.spp import SPP as Model
        self.assert_integrals(Model, kwargs)



    def test_Hernquist(self):
        kwargs = {'sigma0': 1., 'Rs': 1.5}
        from astrofunc.LensingProfiles.hernquist import Hernquist as Model
        self.assert_integrals(Model, kwargs)

    """

if __name__ == '__main__':
    pytest.main("-k TestLensModel")