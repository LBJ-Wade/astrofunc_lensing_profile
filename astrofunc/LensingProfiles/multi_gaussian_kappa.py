import numpy as np
from astrofunc.LensingProfiles.gaussian_kappa import GaussianKappa


class MultiGaussian_kappa(object):
    """

    """

    def __init__(self):
        self.gaussian_kappa = GaussianKappa()

    def function(self, x, y, amps, sigmas, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amps:
        :param sigmas:
        :param center_x:
        :param center_y:
        :return:
        """
        f_ = np.zeros_like(x)
        for i in range(len(amps)):
            f_ += self.gaussian_kappa.function(x, y, amp=amps[i], sigma_x=sigmas[i], sigma_y=sigmas[i],
                                               center_x=center_x, center_y=center_y)
        return f_

    def derivatives(self, x, y, amps, sigmas, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amps:
        :param sigmas:
        :param center_x:
        :param center_y:
        :return:
        """
        f_x, f_y = np.zeros_like(x), np.zeros_like(x)
        for i in range(len(amps)):
            f_x_i, f_y_i = self.gaussian_kappa.derivatives(x, y, amp=amps[i], sigma_x=sigmas[i], sigma_y=sigmas[i],
                                                           center_x=center_x, center_y=center_y)
            f_x += f_x_i
            f_y += f_y_i
        return f_x, f_y

    def hessian(self, x, y, amps, sigmas, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amps:
        :param sigmas:
        :param center_x:
        :param center_y:
        :return:
        """
        f_xx, f_yy, f_xy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for i in range(len(amps)):
            f_xx_i, f_yy_i, f_xy_i = self.gaussian_kappa.hessian(x, y, amp=amps[i], sigma_x=sigmas[i],
                                                                 sigma_y=sigmas[i], center_x=center_x,
                                                                 center_y=center_y)
            f_xx += f_xx_i
            f_yy += f_yy_i
            f_xy += f_xy_i
        return f_xx, f_yy, f_xy

    def all(self, x, y, amps, sigmas, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amps:
        :param sigmas:
        :param center_x:
        :param center_y:
        :return:
        """
        f_ = np.zeros_like(x)
        f_x, f_y = np.zeros_like(x), np.zeros_like(x)
        f_xx, f_yy, f_xy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for i in range(len(amps)):
            f_ += self.gaussian_kappa.function(x, y, amp=amps[i], sigma_x=sigmas[i], sigma_y=sigmas[i],
                                               center_x=center_x, center_y=center_y)
            f_x_i, f_y_i = self.gaussian_kappa.derivatives(x, y, amp=amps[i], sigma_x=sigmas[i], sigma_y=sigmas[i],
                                                           center_x=center_x, center_y=center_y)
            f_x += f_x_i
            f_y += f_y_i
            f_xx_i, f_yy_i, f_xy_i = self.gaussian_kappa.hessian(x, y, amp=amps[i], sigma_x=sigmas[i],
                                                                 sigma_y=sigmas[i], center_x=center_x,
                                                                 center_y=center_y)
            f_xx += f_xx_i
            f_yy += f_yy_i
            f_xy += f_xy_i
        return f_xx, f_yy, f_xy

    def density(self, r, amps, sigmas):
        """

        :param r:
        :param amps:
        :param sigmas:
        :return:
        """
        d_ = np.zeros_like(r)
        for i in range(len(amps)):
            d_ += self.gaussian_kappa.density(r, amps[i], sigmas[i], sigmas[i])
        return d_

    def density_2d(self, x, y, amps, sigmas, center_x=0, center_y=0):
        """

        :param R:
        :param am:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        d_3d = np.zeros_like(x)
        for i in range(len(amps)):
            d_3d += self.gaussian_kappa.density_2d(x, y, amps[i], sigmas[i], sigmas[i], center_x, center_y)
        return d_3d