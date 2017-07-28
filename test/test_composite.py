from astrofunc.LensingProfiles.composite_sersic_nfw import CompositeSersicNFW
from astrofunc.LensingProfiles.sersic import Sersic
from astrofunc.LensingProfiles.nfw_ellipse import NFW_ELLIPSE


class TestMassAngleConversion(object):
    """
    test angular to mass unit conversions
    """
    def setup(self):
        self.composite = CompositeSersicNFW()
        self.sersic = Sersic()
        self.nfw = NFW_ELLIPSE()

    def test_convert(self):
        theta_E = 1.
        mass_light = 1/2.
        Rs = 5.
        n_sersic = 2.
        r_eff = 0.7
        theta_Rs, k_eff = self.composite.convert_mass(theta_E, mass_light, Rs, n_sersic, r_eff)

        alpha_E_sersic, _ = self.sersic.derivatives(theta_E, 0, n_sersic, r_eff, k_eff=1)
        alpha_E_nfw, _ = self.nfw.derivatives(theta_E, 0, Rs, theta_Rs=1, q=1, phi_G=0)
        f_xx_s, f_yy_s, _ = self.sersic.hessian(r_eff, 0, n_sersic, r_eff, k_eff=1)
        f_xx_n, f_yy_n, _ = self.nfw.hessian(r_eff, 0, Rs, theta_Rs=1, q=1, phi_G=0)
        kappa_eff_sersic = (f_xx_s + f_yy_s) / 2.
        kappa_eff_nfw = (f_xx_n + f_yy_n) / 2.
        a = theta_Rs * alpha_E_nfw + k_eff * alpha_E_sersic
        b = theta_Rs * kappa_eff_nfw / (k_eff * kappa_eff_sersic)
        assert a == theta_E
        assert b == mass_light




if __name__ == '__main__':
    pytest.main()