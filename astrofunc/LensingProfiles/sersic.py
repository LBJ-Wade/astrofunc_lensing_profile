__author__ = 'sibirrer'
#this file contains a class to make a gaussian

import numpy as np
import scipy.special as special
import astrofunc.util as util
import astrofunc.LensingProfiles.calc_util as calc_util
from astrofunc.LightProfiles.sersic import SersicUtil


class Sersic(SersicUtil):
    """
    this class contains functions to evaluate a Sersic mass profile: https://arxiv.org/pdf/astro-ph/0311559.pdf
    """
    _s = 0.000001

    def function(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """
        returns Gaussian
        """
        n = n_sersic
        x_red = self._x_reduced(x, y, n, r_eff, center_x, center_y)
        b = self.b_n(n)
        #hyper2f2_b = util.hyper2F2_array(2*n, 2*n, 1+2*n, 1+2*n, -b)
        hyper2f2_bx = util.hyper2F2_array(2*n, 2*n, 1+2*n, 1+2*n, -b*x_red)
        f_eff = np.exp(b)*r_eff**2/2.*k_eff# * hyper2f2_b
        f_ = f_eff * x_red**(2*n) * hyper2f2_bx# / hyper2f2_b
        return f_

    def derivatives(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s
        alpha = self.alpha_abs(x, y, n_sersic, r_eff, k_eff, center_x, center_y)
        f_x = alpha * x_ / r
        f_y = alpha * y_ / r
        return f_x, f_y

    def hessian(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        d_alpha_dr = -self.d_alpha_dr(x, y, n_sersic, r_eff, k_eff, center_x, center_y)
        n = n_sersic
        x_red = self._x_reduced(x, y, n_sersic, r_eff, center_x, center_y)
        b = self.b_n(n_sersic)
        a_eff = self._alpha_eff(r_eff, n_sersic, k_eff)
        alpha = -2. * a_eff * x_red ** (-n) * (1 - special.gammainc(2 * n, b * x_red) / special.gamma(2 * n))

        f_xx = d_alpha_dr * calc_util.d_r_dx(x_, y_) * x_/r + alpha * calc_util.d_x_diffr_dx(x_, y_)
        f_yy = d_alpha_dr * calc_util.d_r_dy(x_, y_) * y_/r + alpha * calc_util.d_y_diffr_dy(x_, y_)
        f_xy = d_alpha_dr * calc_util.d_r_dy(x_, y_) * x_/r + alpha * calc_util.d_x_diffr_dy(x_, y_)
        return f_xx, f_yy, f_xy

    def all(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """
        returns f,f_x,f_y,f_xx, f_yy, f_xy
        """
        f_ = self.function(x, y, n_sersic, r_eff, k_eff, center_x, center_y)
        f_x, f_y = self.derivatives(x, y, n_sersic, r_eff, k_eff, center_x, center_y)
        f_xx, f_yy, f_xy = self.hessian(x, y, n_sersic, r_eff, k_eff, center_x, center_y)
        return f_, f_x, f_y, f_xx, f_yy, f_xy

    def _x_reduced(self, x, y, n_sersic, r_eff, center_x, center_y):
        """
        coordinate transform to normalized radius
        :param x:
        :param y:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s
        x_reduced = (r/r_eff)**(1./n_sersic)
        return x_reduced

    def _alpha_eff(self, r_eff, n_sersic, k_eff):
        """
        deflection angle at r_eff
        :param r_eff:
        :param n_sersic:
        :param k_eff:
        :return:
        """
        b = self.b_n(n_sersic)
        alpha_eff = n_sersic * r_eff * k_eff * b**(-2*n_sersic) * np.exp(b) * special.gamma(2*n_sersic)
        return alpha_eff

    def alpha_abs(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param n_sersic:
        :param r_eff:
        :param k_eff:
        :param center_x:
        :param center_y:
        :return:
        """
        n = n_sersic
        x_red = self._x_reduced(x, y, n_sersic, r_eff, center_x, center_y)
        b = self.b_n(n_sersic)
        a_eff = self._alpha_eff(r_eff, n_sersic, k_eff)
        alpha = 2. * a_eff * x_red**(-n) * (special.gammainc(2*n, b*x_red) / special.gamma(2*n))
        return alpha

    def d_alpha_dr(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param n_sersic:
        :param r_eff:
        :param k_eff:
        :param center_x:
        :param center_y:
        :return:
        """
        n = n_sersic
        x_red = self._x_reduced(x, y, n, r_eff, center_x, center_y)
        b = self.b_n(n)
        a_eff = self._alpha_eff(r_eff, n_sersic, k_eff)
        f_r_1 = 2 * a_eff / r_eff * x_red**(-2*n) * (1 - special.gammainc(2*n, b*x_red) / special.gamma(2*n))  # equation 21
        f_r_2 = 2 * a_eff / r_eff / n * b**(2*n) / special.gamma(2*n) * np.exp(-b*x_red)  # equation 22
        return -(f_r_1 + f_r_2)