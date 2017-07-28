__author__ = 'sibirrer'
#this file contains a class to make a gaussian

import numpy as np
from astrofunc.LensingProfiles.sersic import Sersic


class SersicEllipse(object):
    """
    this class contains functions to evaluate a Sersic mass profile: https://arxiv.org/pdf/astro-ph/0311559.pdf
    """
    def __init__(self):
        self.sersic = Sersic()

    def function(self, x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x=0, center_y=0):
        """
        returns Gaussian
        """
        x_, y_ = self._coord_transf(x, y, q, phi_G, center_x, center_y)
        f_ = self.sersic.function(x_, y_, n_sersic, r_eff, k_eff)
        return f_

    def derivatives(self, x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        e = 1. - q
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        x_, y_ = self._coord_transf(x, y, q, phi_G, center_x, center_y)
        f_x_prim, f_y_prim = self.sersic.derivatives(x_, y_, n_sersic, r_eff, k_eff)
        f_x_prim *= np.sqrt(1 - e)
        f_y_prim *= np.sqrt(1 + e)
        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        # TODO: not exactly correct: see Bolse & Kneib 2002
        x_, y_ = self._coord_transf(x, y, q, phi_G, center_x, center_y)
        f_xx_p, f_yy_p, f_xy_p = self.sersic.hessian(x_, y_, n_sersic, r_eff, k_eff)
        kappa = 1./2. * (f_xx_p + f_yy_p)
        gamma1_p = 1./2 * (f_xx_p - f_yy_p)  # attention on units
        gamma2_p = f_xy_p  # attention on units
        gamma1 = np.cos(2*phi_G)*gamma1_p-np.sin(2*phi_G)*gamma2_p
        gamma2 = +np.sin(2*phi_G)*gamma1_p+np.cos(2*phi_G)*gamma2_p
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def all(self, x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x=0, center_y=0):
        """
        returns f,f_x,f_y,f_xx, f_yy, f_xy
        """
        f_ = self.function(x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x, center_y)
        f_x, f_y = self.derivatives(x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x, center_y)
        f_xx, f_yy, f_xy = self.hessian(x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x, center_y)
        return f_, f_x, f_y, f_xx, f_yy, f_xy

    def _coord_transf(self, x, y, q, phi_G, center_x, center_y):
        """

        :param x:
        :param y:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = 1 - q
        x_ = (cos_phi * x_shift + sin_phi * y_shift) * np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) * np.sqrt(1 + e)
        return x_, y_