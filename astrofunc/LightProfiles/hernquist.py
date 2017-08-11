

class Hernquist(object):
    """
    class for pseudo Jaffe lens light (2d projected light/mass distribution
    """
    def __init__(self):
        from astrofunc.LensingProfiles.hernquist import Hernquist as Hernquist_lens
        self.lens = Hernquist_lens()

    def function(self, x, y, sigma0, Rs, center_x=0, center_y=0):
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
        rho0 = self.lens.sigma2rho(sigma0, Rs)
        return self.lens.density_2d(x, y, rho0, Rs, center_x, center_y)

    def light_3d(self, r, sigma0, Rs, center_x=0, center_y=0):
        """

        :param y:
        :param sigma0:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        rho0 = self.lens.sigma2rho(sigma0, Rs)
        return self.lens.density(r, rho0, Rs)


class Hernquist_Ellipse(object):
    """
    class for elliptical pseudo Jaffe lens light (2d projected light/mass distribution
    """
    def __init__(self):
        from astrofunc.LensingProfiles.hernquist_ellipse import Hernquist_Ellipse as Hernquist_lens
        self.lens = Hernquist_lens()

    def function(self, x, y, sigma0, Rs, q, phi_G, center_x=0, center_y=0):
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
        f_xx, f_yy, _ = self.lens.hessian(x, y, sigma0, Rs, q, phi_G, center_x, center_y)
        return 1./2. * (f_xx + f_yy)