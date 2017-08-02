from astrofunc.LensingProfiles.p_jaffe import PJaffe as PJaffe_lens


class PJaffe(object):
    """
    class for pseudo Jaffe lens light (2d projected light/mass distribution
    """
    def __init__(self):
        self.p_Jaffe = PJaffe_lens()

    def function(self, x, y, sigma0, a, s, center_x, center_y):
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
        return self.p_Jaffe.density_2d(x, y, sigma0, a, s, center_x, center_y)