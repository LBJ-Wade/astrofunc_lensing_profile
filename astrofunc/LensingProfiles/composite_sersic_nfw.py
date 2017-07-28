from astrofunc.LensingProfiles.sersic import Sersic
from astrofunc.LensingProfiles.nfw_ellipse import NFW_ELLIPSE


class CompositeSersicNFW(object):
    """
    class for a composite model (Sersic and NFW profile combined)
    with joint center and parameterization of Einstein radius
    """
    def __init__(self):
        self.sersic = Sersic()
        self.nfw = NFW_ELLIPSE()

    def function(self, x, y, theta_E, mass_light, Rs, q, phi_G, n_sersic, r_eff, center_x=0, center_y=0):
        """

        :param theta_E:
        :param mass_light:
        :param Rs:
        :param q:
        :param phi_G:
        :param n_sersic:
        :param r_eff:
        :param center_x:
        :param center_y:
        :return:
        """
        theta_Rs, k_eff = self.convert_mass(theta_E, mass_light, Rs, n_sersic, r_eff)
        f_s = self.sersic.function(x, y, n_sersic, r_eff, k_eff, center_x, center_y)
        f_nfw = self.nfw.function(x, y, Rs, theta_Rs,q, phi_G, center_x, center_y)
        return f_s + f_nfw

    def derivatives(self, x, y, theta_E, mass_light, Rs, q, phi_G, n_sersic, r_eff, center_x=0, center_y=0):
        """

        :param theta_E:
        :param mass_light:
        :param Rs:
        :param q:
        :param phi_G:
        :param n_sersic:
        :param r_eff:
        :param center_x:
        :param center_y:
        :return:
        """
        theta_Rs, k_eff = self.convert_mass(theta_E, mass_light, Rs, n_sersic, r_eff)
        f_x_s, f_y_s = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff, center_x, center_y)
        f_x_nfw, f_y_nfw = self.nfw.derivatives(x, y, Rs, theta_Rs,q, phi_G, center_x, center_y)
        return f_x_s + f_x_nfw, f_y_s + f_y_nfw

    def hessian(self, x, y, theta_E, mass_light, Rs, q, phi_G, n_sersic, r_eff, center_x=0, center_y=0):
        """

        :param theta_E:
        :param mass_light:
        :param Rs:
        :param q:
        :param phi_G:
        :param n_sersic:
        :param r_eff:
        :param center_x:
        :param center_y:
        :return:
        """
        theta_Rs, k_eff = self.convert_mass(theta_E, mass_light, Rs, n_sersic, r_eff)
        f_xx_s, f_yy_s, f_xy_s = self.sersic.hessian(x, y, n_sersic, r_eff, k_eff, center_x, center_y)
        f_xx_nfw, f_yy_nfw, f_xy_nfw = self.nfw.hessian(x, y, Rs, theta_Rs,q, phi_G, center_x, center_y)
        return f_xx_s + f_xx_nfw, f_yy_s + f_yy_nfw, f_xy_s + f_xy_nfw

    def all(self, x, y, theta_E, mass_light, Rs, q, phi_G, n_sersic, r_eff, center_x=0, center_y=0):
        """

        :param theta_E:
        :param mass_light:
        :param Rs:
        :param q:
        :param phi_G:
        :param n_sersic:
        :param r_eff:
        :param center_x:
        :param center_y:
        :return:
        """
        theta_Rs, k_eff = self.convert_mass(theta_E, mass_light, Rs, n_sersic, r_eff)
        f_s, f_x_s, f_y_s, f_xx_s, f_yy_s, f_xy_s = self.sersic.all(x, y, n_sersic, r_eff, k_eff, center_x, center_y)
        f_nfw, f_x_nfw, f_y_nfw, f_xx_nfw, f_yy_nfw, f_xy_nfw = self.nfw.all(x, y, Rs, theta_Rs,q, phi_G, center_x, center_y)
        return f_s + f_nfw, f_x_s + f_x_nfw, f_y_s + f_y_nfw, f_xx_s + f_xx_nfw, f_yy_s + f_yy_nfw, f_xy_s + f_xy_nfw

    def convert_mass(self, theta_E, mass_light, Rs, n_sersic, r_eff):
        """
        convert global parameters theta_E and mass_light to specific ones theta_Rs and k_eff
        :param theta_E:
        :param mass_light:
        :param Rs:
        :param n_sersic:
        :param r_eff:
        :return:
        """
        alpha_E_sersic, _ = self.sersic.derivatives(theta_E, 0, n_sersic, r_eff, k_eff=1)
        alpha_E_nfw, _ = self.nfw.derivatives(theta_E, 0, Rs, theta_Rs=1, q=1, phi_G=0)
        f_xx_s, f_yy_s, _ = self.sersic.hessian(r_eff, 0, n_sersic, r_eff, k_eff=1)
        f_xx_n, f_yy_n, _ = self.nfw.hessian(r_eff, 0, Rs, theta_Rs=1, q=1, phi_G=0)
        kappa_eff_sersic = (f_xx_s + f_yy_s) / 2.
        kappa_eff_nfw = (f_xx_n + f_yy_n) / 2.
        # equations must satisfy:
        # theta_Rs * alpha_E_nfw + k_eff * alpha_E_sersic = theta_E
        # theta_Rs * kappa_eff_nfw / (k_eff * kappa_eff_sersic) = mass_light
        k_eff = theta_E * kappa_eff_nfw / (alpha_E_sersic*kappa_eff_nfw+mass_light*alpha_E_nfw*kappa_eff_sersic)
        theta_Rs = (theta_E - k_eff*alpha_E_sersic)/alpha_E_nfw
        return theta_Rs, k_eff