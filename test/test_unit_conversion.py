__author__ = 'sibirrer'


from astrofunc.LensingProfiles.spp import SPP

import numpy as np
import numpy.testing as npt
import astrofunc.constants as const
import pytest


class TestSIS(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.spp = SPP()
        self.rho0_kgm3 = const.rho_c(0.7) * 10000  # density in kg/m^3 at 1 arc_sec
        self.gamma = 2.  # power-law slope of SIS
        self.D_d = 1000.  # units of Mpc
        self.D_s = 1500.  # units of Mpc
        self.D_ds = 800.  # units of Mpc


    def test_spp_mass_enclosed(self):

        arc_sec_m = const.arcsec * self.D_d * const.Mpc
        mass_3d = self.spp.mass_3d(1., self.rho0_kgm3, self.gamma) * arc_sec_m**3
        density_2d = self.spp.density_2d(1, 0, self.rho0_kgm3, self.gamma)
        Epsion_crit = const.c**2 * self.D_s / (4*np.pi*const.G * self.D_ds * self.D_d) * (arc_sec_m/const.Mpc)**2  # in units of arcsec ^2
        print density_2d*arc_sec_m**2/Epsion_crit, 'sigma2d/epsilon_crit'

        mass_3d_msol = mass_3d / const.M_sun

        theta_E = self.spp.rho2theta(self.rho0_kgm3 * arc_sec_m**2/Epsion_crit, self.gamma)
        mass_3d_lens = self.spp.mass_3d_lens(1., theta_E, self.gamma) / arc_sec_m**2 * Epsion_crit * arc_sec_m**3
        mass_3d_lens_new = self.spp.mass_3d_lens(1., theta_E, self.gamma) * const.arcsec**3 * self.D_d**2 * const.Mpc * const.c**2 * self.D_s / (4*np.pi*const.G * self.D_ds)
        mass_3d_lens_msol = mass_3d_lens / const.M_sun
        print theta_E, 'theta_E'
        print mass_3d_lens, 'mass_3d_lens'
        npt.assert_almost_equal(np.log10(mass_3d_msol), np.log10(mass_3d_lens_msol), decimal=7)
        npt.assert_almost_equal(np.log10(mass_3d_lens_new), np.log10(mass_3d_lens), decimal=7)


if __name__ == '__main__':
    pytest.main()