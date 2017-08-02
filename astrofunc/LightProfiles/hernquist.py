from astrofunc.LensingProfiles.hernquist import Hernquist as Hernquist_lens


class Hernquist(object):
    """
    class for pseudo Jaffe lens light (2d projected light/mass distribution
    """
    def __init__(self):
        self.lens = Hernquist_lens()

    def function(self, x, y, sigma0, Rs, center_x, center_y):
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
        return self.lens.density_2d(x, y, sigma0, Rs, center_x, center_y)