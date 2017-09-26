import numpy as np


class Gaussian(object):
    """
    class for Gaussian light profile
    """
    def __init__(self):
        pass

    def function(self, x, y, amp, sigma, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        c = amp / (2 * np.pi * sigma**2)
        R2 = (x - center_x) ** 2 + (y - center_y) ** 2
        return c * np.exp(-(R2 / float(sigma)**2)/ 2.)

    def light_3d(self, r, amp, sigma):
        """

        :param y:
        :param sigma0:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        amp3d = amp / sigma * np.sqrt(np.pi/2)
        sigma3d = sigma
        return self.function(r, 0, amp3d, sigma3d)


class MultiGaussian(object):
    """
    class for elliptical pseudo Jaffe lens light (2d projected light/mass distribution
    """
    def __init__(self):
        self.gaussian = Gaussian()

    def function(self, x, y, amp, sigma, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        f_ = np.zeros_like(x)
        for i in range(len(amp)):
            f_ += self.gaussian.function(x, y, amp[i], sigma[i], center_x, center_y)
        return f_

    def light_3d(self, r, amp, sigma):
        """

        :param y:
        :param sigma0:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        f_ = np.zeros_like(r)
        for i in range(len(amp)):
            f_ += self.gaussian.light_3d(r, amp[i], sigma[i])
        return f_
