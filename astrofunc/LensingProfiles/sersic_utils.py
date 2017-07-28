import scipy.special as special
import numpy as np


class SersicUtil(object):

    _s = 0.000001

    def k_bn(self, n, Re):
        """
        returns normalisation of the sersic profile such that Re is the half light radius given n_sersic slope
        """
        bn = self.b_n(n)
        k = bn*Re**(-1./n)
        return k, bn

    def k_Re(self, n, k):
        """

        """
        bn = self.b_n(n)
        Re = (bn/k)**n
        return Re

    def b_n(self, n):
        """
        b(n) computation
        :param n:
        :return:
        """
        bn = 1.9992 * n - 0.3271
        return bn

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
        return -alpha_eff

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
        #alpha = 2. * a_eff * x_red**(-n) * (1 - special.gammainc(2*n, b*x_red) / special.gamma(2*n))
        alpha = 2. * a_eff * x_red ** (-n) * (special.gammainc(2 * n, b * x_red))
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
        p = self._p(x, y, n_sersic, r_eff, k_eff, center_x, center_y)
        q = self._q(x, y, n_sersic, r_eff, k_eff, center_x, center_y)
        return -(p + q)

    def _p(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """
        equation (21) in paper
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
        #f_r_1 = 2 * a_eff / r_eff * x_red**(-2*n) * (1 - special.gammainc(2*n, b*x_red) / special.gamma(2*n))  # equation 21
        f_r_1 = 2 * a_eff / r_eff * x_red**(-2*n) * (special.gammainc(2*n, b*x_red))  # equation 21

        return f_r_1

    def _q(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """
        equation (22) in paper
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
        f_r_2 = 2 * a_eff / r_eff / n * b**(2*n) / special.gamma(2*n) * np.exp(-b*x_red)  # equation 22
        return f_r_2