


class PJaffe(object):
    """
    class for pseudo Jaffe lens light (2d projected light/mass distribution)
    """
    def __init__(self):
        from astrofunc.LensingProfiles.p_jaffe import PJaffe as PJaffe_lens
        self.p_Jaffe = PJaffe_lens()

    def function(self, x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
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
        return self.p_Jaffe.density_2d(x, y, sigma0, Ra, Rs, center_x, center_y)

class PJaffe_Ellipse(object):
    """
    calss for elliptical pseudo Jaffe lens light
    """
    def __init__(self):
        from astrofunc.LensingProfiles.p_jaffe_ellipse import PJaffe_Ellipse as PJaffe_lens
        self.p_Jaffe = PJaffe_lens()

    def function(self, x, y, sigma0, Ra, Rs, q, phi_G, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0:
        :param Ra:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        f_xx, f_yy, _ = self.p_Jaffe.hessian(x, y, sigma0, Ra, Rs, q, phi_G, center_x, center_y)
        return 1./2. * (f_xx + f_yy)
