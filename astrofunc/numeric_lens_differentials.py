from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'


class NumericLens(object):
    """
    this class computes numerical differentials of lens model quantities
    """
    def __init__(self, lensModel, diff):
        """

        :param lensModel:
        :param diff:
        """
        self.lensModel = lensModel
        self._diff = diff

    def kappa(self, x, y, kwargs):
        """
        computes the convergence
        :return: kappa
        """
        f_xx, f_yy, f_xy = self.hessian(x, y, kwargs)
        kappa = 1./2 * (f_xx + f_yy)
        return kappa

    def gamma(self, x, y, kwargs):
        """
        computes the shear
        :return: gamma1, gamma2
        """
        f_xx, f_yy, f_xy = self.hessian(x, y, kwargs)
        gamma1 = 1./2 * (f_yy - f_xx)
        gamma2 = f_xy
        return gamma1, gamma2

    def magnification(self, x, y, kwargs):
        """
        computes the magnification
        :return: potential
        """
        f_xx, f_yy, f_xy = self.hessian(x, y, kwargs)
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy**2
        return 1/det_A

    def derivatives(self, x, y, kwargs):
        """

        :param x:
        :param y:
        :param kwargs:
        :return:
        """
        f_ = self.lensModel.function(x, y, **kwargs)
        diff = self._diff
        f_dx = self.lensModel.function(x + diff, y, **kwargs)
        f_dy = self.lensModel.function(x, y + diff, **kwargs)
        f_x = (f_dx - f_)/diff
        f_y = (f_dy - f_)/diff
        return f_x, f_y

    def hessian(self, x, y, kwargs):
        """
        computes the differentials f_xx, f_yy, f_xy from f_x and f_y
        :return: f_xx, f_xy, f_yx, f_yy
        """
        alpha_ra, alpha_dec = self.lensModel.derivatives(x, y, **kwargs)
        diff = self._diff
        alpha_ra_dx, alpha_dec_dx = self.lensModel.derivatives(x + diff, y, **kwargs)
        alpha_ra_dy, alpha_dec_dy = self.lensModel.derivatives(x, y + diff, **kwargs)

        dalpha_rara = (alpha_ra_dx - alpha_ra)/diff
        dalpha_radec = (alpha_ra_dy - alpha_ra)/diff
        dalpha_decra = (alpha_dec_dx - alpha_dec)/diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec)/diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_yy, f_xy