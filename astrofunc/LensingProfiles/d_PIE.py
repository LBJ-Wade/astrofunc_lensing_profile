import numpy as np


class D_PIE(object):
    """
    class to compute the DUAL PSEUDO ISOTHERMAL ELLIPTICAL MASS DISTRIBUTION
    based on Eliasdottir (2013)
    """

    def density(self, x, y, rho0, a, s,  center_x=0, center_y=0):
        """
        computes the density
        :param x:
        :param y:
        :param rho0:
        :param a:
        :param s:
        :return:
        """
        if a >= s:
            a, s = s, a
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        rho = rho0 / ((1 + (r/a)**2) * (1 + (r/s)**2))
        return rho

    def density_2d(self, x, y, rho0, a, s, center_x, center_y):
        """
        projected density
        :param x:
        :param y:
        :param rho0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        if a >= s:
            a, s = s, a
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        sigma0 = self.rho2sigma(rho0, a, s)
        sigma = sigma0 * a*s/(s-a) * (1/np.sqrt(a**2 + r**2) - 1/np.sqrt(s**2 + r**2))
        return sigma

    def mass_3d(self, r, rho0, a, s):
        """
        mass enclosed a 3d sphere or radius r
        :param r:
        :param a:
        :param s:
        :return:
        """
        m_3d = 4 * np.pi * rho0 * a**2*s**2/(s**2-a**2) * (s*np.arctan(r/s) - a*np.arctan(r/a))
        return m_3d

    def mass_2d(self, r, rho0, a, s):
        """
        mass enclosed projected 2d sphere of radius r
        :param r:
        :param rho0:
        :param a:
        :param s:
        :return:
        """
        sigma0 = self.rho2sigma(rho0, a, s)
        m_2d = 2 * np.pi * sigma0 * a*s/(s-a) * (np.sqrt(a**2 + r**2) - a - np.sqrt(s**2 + r**2) + s)
        return m_2d

    def mass_tot(self, rho0, a, s):
        """
        total mass within the profile
        :param rho0:
        :param a:
        :param s:
        :return:
        """
        sigma0 = self.rho2sigma(rho0, a, s)
        m_tot = 2 * np.pi * sigma0 * a*s
        return m_tot

    def grav_pot(self, x, y, rho0, a, s,  center_x=0, center_y=0):
        """
        gravitational potential (modulo 4 pi G and rho0 in appropriate units)
        :param x:
        :param y:
        :param rho0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        if a >= s:
            a, s = s, a
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        pot = 4 * np.pi * rho0 * a**2*s**2/(s**2-a**2) * (s/r * np.arctan(r/s) - a/r * np.arctan(r/a)
                                                          + 1./2*np.log((s**2 + r**2)/(a**2 + r**2)))
        return pot

    def function(self, x, y, sigma0, a, s,  center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0: sigma0/sigma_crit
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        if a >= s:
            a, s = s, a
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        f_ = -2*sigma0 * a*s/(s-a) * (np.sqrt(s**2+r**2) - np.sqrt(a**2+r**2) + a*np.log(a + np.sqrt(a**2+r**2)) - s*np.log(s + np.sqrt(s**2+r**2)))
        return f_

    def derivatives(self, x, y, sigma0, a, s, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0: sigma0/sigma_crit
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        if a >= s:
            a, s = s, a
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        alpha_r = 2*sigma0 * a*s/(s-a) * self._f_A20(r/a, r/s)
        f_x = alpha_r * x_/r
        f_y = alpha_r * y_/r
        return f_x, f_y

    def hessian(self, x, y, sigma0, a, s,  center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0: sigma0/sigma_crit
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        if a >= s:
            a, s = s, a
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        gamma = sigma0 * a*s/(s-a) * (2*(1./(a + np.sqrt(a**2+r**2)) - 1./(s + np.sqrt(s**2+r**2))) -
                                     (1/np.sqrt(a**2+r**2) - 1/np.sqrt(s**2+r**2)))
        kappa = sigma0 * a*s/(s-a) * (1/np.sqrt(a**2+r**2) - 1/np.sqrt(s**2+r**2))
        sin_2phi = -2*x_*y_/r**2
        cos_2phi = (y_**2 - x_**2)/r**2
        gamma1 = cos_2phi*gamma
        gamma2 = sin_2phi*gamma

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def _f_A20(self, r_a, r_s):
        """
        equation A20 in Eliasdottir (2013)
        :param r_a:
        :param r_s:
        :return:
        """
        return r_a/(1+np.sqrt(1 + r_a**2)) - r_s/(1+np.sqrt(1 + r_s**2))

    def rho2sigma(self, rho0, a, s):
        """
        converts 3d density into 2d projected density parameter
        :param rho0:
        :param a:
        :param s:
        :return:
        """
        return np.pi * rho0 * a*s/(s+a)