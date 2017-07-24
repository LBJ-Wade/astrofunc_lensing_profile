__author__ = 'sibirrer'

#this file contains a class to make a sersic profile

import numpy as np


class SersicUtil(object):

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


class Sersic(SersicUtil):
    """
    this class contains functions to evaluate an spherical Sersic function
    """
    def function(self, x, y, I0_sersic, R_sersic, n_sersic, center_x, center_y, smoothing=0.02):
        """
        returns Sersic profile
        """
        if n_sersic < 0.2:
            n_sersic = 0.2
        if R_sersic < 10.**(-6):
            R_sersic = 10.**(-6)
        x_shift = x - center_x
        y_shift = y - center_y
        R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
        if isinstance(R, int) or isinstance(R, float):
            R = max(smoothing, R)
        else:
            R[R < smoothing] = smoothing
        _, bn = self.k_bn(n_sersic, R_sersic)
        R_frac = R/R_sersic
        #R_frac = R_frac.astype(np.float32)
        if isinstance(R, int) or isinstance(R, float):
            if R_frac > 20:
                result = 0
            else:
                exponent = -bn*(R_frac**(1./n_sersic)-1.)
                result = I0_sersic * np.exp(exponent)
        else:
            R_frac_real = R_frac[R_frac <= 20]
            exponent = -bn*(R_frac_real**(1./n_sersic)-1.)
            result = np.zeros_like(R)
            result[R_frac <= 20] = I0_sersic * np.exp(exponent)
        return result


class Sersic_elliptic(SersicUtil):
    """
    this class contains functions to evaluate an elliptical Sersic function
    """

    def function(self, x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y, smoothing=0.02):
        """
        returns Sersic profile
        """
        if n_sersic < 0.2:
            n_sersic = 0.2
        if R_sersic < 10.**(-6):
            R_sersic = 10.**(-6)
        x_shift = x - center_x
        y_shift = y - center_y

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1 = cos_phi*x_shift+sin_phi*y_shift
        xt2 = -sin_phi*x_shift+cos_phi*y_shift
        xt2difq2 = xt2/(q*q)
        R_ = np.sqrt(xt1*xt1+xt2*xt2difq2)
        if isinstance(R_, int) or isinstance(R_, float):
            R_ = max(smoothing, R_)
        else:
            R_[R_ < smoothing] = smoothing
        k, bn = self.k_bn(n_sersic, R_sersic)
        R_frac = R_/R_sersic
        R_frac = R_frac.astype(np.float32)
        if isinstance(R_, int) or isinstance(R_, float):
            if R_frac > 20:
                result = 0
            else:
                exponent = -bn*(R_frac**(1./n_sersic)-1.)
                result = I0_sersic * np.exp(exponent)
        else:
            R_frac_real = R_frac[R_frac <= 20]
            exponent = -bn*(R_frac_real**(1./n_sersic)-1.)
            result = np.zeros_like(R_)
            result[R_frac <= 20] = I0_sersic * np.exp(exponent)
        return result


class CoreSersic(SersicUtil):
    """
    this class contains the Core-Sersic function introduced by e.g Trujillo et al. 2004
    """

    def function(self, x, y, I0_sersic, R_sersic, Re, n_sersic, gamma, phi_G, q, center_x, center_y, alpha=3., smoothing=0.02):
        """
        returns Core-Sersic function
        """
        Rb = R_sersic
        x_shift = x - center_x
        y_shift = y - center_y

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1 = cos_phi*x_shift+sin_phi*y_shift
        xt2 = -sin_phi*x_shift+cos_phi*y_shift
        xt2difq2 = xt2/(q*q)
        R_ = np.sqrt(xt1*xt1+xt2*xt2difq2)
        R_ = R_.astype(np.float32)
        if isinstance(R_, int) or isinstance(R_, float):
            R_ = max(smoothing, R_)
        else:
            R_[R_ < smoothing] = smoothing
        if isinstance(R_, int) or isinstance(R_, float):
            R = max(smoothing, R_)
        else:
            R=np.empty_like(R_)
            _R = R_[R_ > smoothing]  #in the SIS regime
            R[R_ <= smoothing] = smoothing
            R[R_ > smoothing] = _R

        k, bn = self.k_bn(n_sersic, Re)
        return I0_sersic * (1 + (Rb/R)**alpha)**(gamma/alpha) * np.exp(-bn*(((R**alpha+Rb**alpha)/Re**alpha)**(1./(alpha*n_sersic))-1.))

class DoubleSersic(object):
    """
    this class contains functions to evaluate an elliptical and a spherical sersic function at once
    """
    def __init__(self):
        self.sersic = Sersic_elliptic()
        self.sersic_ellipse = Sersic_elliptic()

    def function(self, x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y, I0_2, R_2, n_2, smoothing=0.02):
        ellipse = self.sersic_ellipse.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y, smoothing)
        spherical = self.sersic.function(x, y, I0_2, R_2, n_2, phi_G, q, center_x, center_y, smoothing)
        return ellipse + spherical

    def function_split(self, x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y, I0_2, R_2, n_2, smoothing=0.02):
        ellipse = self.sersic_ellipse.function(x, y, I0_sersic, R_sersic, n_sersic, phi_G, q, center_x, center_y, smoothing)
        spherical = self.sersic.function(x, y, I0_2, R_2, n_2, phi_G, q, center_x, center_y, smoothing)
        return ellipse, spherical


class DoubleCoreSersic(object):
    """
    this class contains functions to evaluate an elliptical core sersic and a spherical sersic function at once
    """
    def __init__(self):
        self.sersic = Sersic_elliptic()
        self.sersic_core = CoreSersic()

    def function(self, x, y, I0_sersic, Re, R_sersic, n_sersic, gamma, phi_G, q, center_x, center_y, I0_2, R_2, n_2, smoothing=0.02):
        core_ellipse = self.sersic_core.function(x, y, I0_sersic, Re, R_sersic, n_sersic, gamma, phi_G, q, center_x, center_y, smoothing)
        spherical = self.sersic.function(x, y, I0_2, R_2, n_2, phi_G, q, center_x, center_y, smoothing)
        return core_ellipse + spherical

    def function_split(self, x, y, I0_sersic, Re, R_sersic, n_sersic, gamma, phi_G, q, center_x, center_y, I0_2, R_2, n_2, smoothing=0.02):
        core_ellipse = self.sersic_core.function(x, y, I0_sersic, Re, R_sersic, n_sersic, gamma, phi_G, q, center_x, center_y, smoothing)
        spherical = self.sersic.function(x, y, I0_2, R_2, n_2, phi_G, q, center_x, center_y, smoothing)
        return core_ellipse, spherical


class BuldgeDisk(object):
    """
    this class handles a buldge-to-disk decomposition model
    """
    def __init__(self):
        self.sersic = Sersic_elliptic()
        self.n_buldge = 4
        self.n_disk = 1

    def function(self, x, y, I0_b, R_b, phi_G_b, q_b, I0_d, R_d, phi_G_d, q_d, center_x, center_y, smoothing=0.03):
        buldge = self.sersic.function(x, y, I0_b, R_b, self.n_buldge, phi_G_b, q_b, center_x, center_y, smoothing)
        disk = self.sersic.function(x, y, I0_d, R_d, self.n_disk, phi_G_d, q_d, center_x, center_y, smoothing)
        return buldge + disk

    def function_split(self, x, y, I0_b, R_b, phi_G_b, q_b, I0_d, R_d, phi_G_d, q_d, center_x, center_y, smoothing=0.03):
        buldge = self.sersic.function(x, y, I0_b, R_b, self.n_buldge, phi_G_b, q_b, center_x, center_y, smoothing)
        disk = self.sersic.function(x, y, I0_d, R_d, self.n_disk, phi_G_d, q_d, center_x, center_y, smoothing)
        return buldge, disk