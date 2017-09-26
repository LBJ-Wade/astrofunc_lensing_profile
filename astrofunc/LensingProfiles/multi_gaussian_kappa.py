import numpy as np
from astrofunc.LensingProfiles.gaussian_kappa import GaussianKappa


class MultiGaussian_kappa(object):
    """

    """

    def __init__(self):
        self.gaussian_kappa = GaussianKappa()

    def function(self, x, y, amp, sigma, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        f_ = np.zeros_like(x)
        for i in range(len(amp)):
            f_ += self.gaussian_kappa.function(x, y, amp=amp[i], sigma_x=sigma[i], sigma_y=sigma[i],
                                               center_x=center_x, center_y=center_y)
        return f_

    def derivatives(self, x, y, amp, sigma, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        f_x, f_y = np.zeros_like(x), np.zeros_like(x)
        for i in range(len(amp)):
            f_x_i, f_y_i = self.gaussian_kappa.derivatives(x, y, amp=amp[i], sigma_x=sigma[i], sigma_y=sigma[i],
                                                           center_x=center_x, center_y=center_y)
            f_x += f_x_i
            f_y += f_y_i
        return f_x, f_y

    def hessian(self, x, y, amp, sigma, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        f_xx, f_yy, f_xy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for i in range(len(amp)):
            f_xx_i, f_yy_i, f_xy_i = self.gaussian_kappa.hessian(x, y, amp=amp[i], sigma_x=sigma[i],
                                                                 sigma_y=sigma[i], center_x=center_x,
                                                                 center_y=center_y)
            f_xx += f_xx_i
            f_yy += f_yy_i
            f_xy += f_xy_i
        return f_xx, f_yy, f_xy

    def all(self, x, y, amp, sigma, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        f_ = np.zeros_like(x)
        f_x, f_y = np.zeros_like(x), np.zeros_like(x)
        f_xx, f_yy, f_xy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for i in range(len(amp)):
            f_ += self.gaussian_kappa.function(x, y, amp=amp[i], sigma_x=sigma[i], sigma_y=sigma[i],
                                               center_x=center_x, center_y=center_y)
            f_x_i, f_y_i = self.gaussian_kappa.derivatives(x, y, amp=amp[i], sigma_x=sigma[i], sigma_y=sigma[i],
                                                           center_x=center_x, center_y=center_y)
            f_x += f_x_i
            f_y += f_y_i
            f_xx_i, f_yy_i, f_xy_i = self.gaussian_kappa.hessian(x, y, amp=amp[i], sigma_x=sigma[i],
                                                                 sigma_y=sigma[i], center_x=center_x,
                                                                 center_y=center_y)
            f_xx += f_xx_i
            f_yy += f_yy_i
            f_xy += f_xy_i
        return f_xx, f_yy, f_xy

    def density(self, r, amp, sigma):
        """

        :param r:
        :param amp:
        :param sigma:
        :return:
        """
        d_ = np.zeros_like(r)
        for i in range(len(amp)):
            d_ += self.gaussian_kappa.density(r, amp[i], sigma[i], sigma[i])
        return d_

    def density_2d(self, x, y, amp, sigma, center_x=0, center_y=0):
        """

        :param R:
        :param am:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        d_3d = np.zeros_like(x)
        for i in range(len(amp)):
            d_3d += self.gaussian_kappa.density_2d(x, y, amp[i], sigma[i], sigma[i], center_x, center_y)
        return d_3d