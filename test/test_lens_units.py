import numpy as np
import numpy.testing as npt
import pytest

from astrofunc.Cosmo.lens_unit_conversion import LensUnits
from astropy.cosmology import WMAP9


class TestLensUnits(object):

    def setup(self):
        z_d = 0.5
        z_s = 2
        self.lensUnits = LensUnits(z_lens=z_d, z_source=z_s, cosmo=WMAP9)

    def test_D_d(self):
        D_d = self.lensUnits.D_d
        assert D_d == 1277.37961573603

    def test_D_s(self):
        D_s = self.lensUnits.D_s
        assert D_s == 1763.9101459409605

    def test_D_ds(self):
        D_ds = self.lensUnits.D_ds
        assert D_ds == 1125.2203380729454

    def test_epsilon_crit(self):
        Epsilon_crit = self.lensUnits.epsilon_crit
        assert Epsilon_crit == 2040180453480701.5

    def test_sis_physical2angle(self):
        v_sigma = 200
        theta_E = self.lensUnits.sis_physical2angle(v_sigma=v_sigma)
        assert theta_E == 0.7358930438688159

    def test_sis_angle2physical(self):
        theta_E = 0.7358930438688159
        v_sigma = self.lensUnits.sis_angle2physical(theta_E)
        npt.assert_almost_equal(v_sigma, 200, decimal=8)

    def test_nfw_physical2angle(self):
        M = 10.**13
        c = 4
        Rs, theta_Rs = self.lensUnits.nfw_physical2angle(M, c)
        print(Rs, theta_Rs)
        assert theta_Rs == 0.6442867161978858
        assert Rs == 5.784764319257965

    def test_nfw_angle2physical(self):
        M = 10.**13
        c = 4.
        rho0, Rs, r200 = self.lensUnits.nfwParam_physical(M, c)
        Rs_angle, theta_Rs = self.lensUnits.nfw_physical2angle(M, c)
        rho0_out, Rs_out, c_out, r200_out, M_out = self.lensUnits.nfw_angle2physical(Rs_angle, theta_Rs)
        print(rho0_out, Rs_out, c_out, r200_out, M_out)
        npt.assert_almost_equal(rho0, rho0_out, decimal=-1)
        npt.assert_almost_equal(Rs, Rs_out, decimal=9)
        npt.assert_almost_equal(c, c_out, decimal=9)
        assert np.log10(M) == np.log10(M_out)


if __name__ == '__main__':
    pytest.main()