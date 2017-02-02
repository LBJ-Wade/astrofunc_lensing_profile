__author__ = 'sibirrer'


class SPEP_Shapelets(object):
    """
    combination of polar shapelets and SPEP profile
    """
    def __init__(self, type='SPEP'):
        from astrofunc.LensingProfiles.ellipse import Ellipse
        self.spep = Ellipse(type)
        from astrofunc.LensingProfiles.shapelet_pot_2 import CartShapelets
        self.cartShapelets = CartShapelets()

    def function(self, x, y, phi_E, gamma, q, phi_G, coeffs, beta, center_x=0, center_y=0, center_x_shape=0, center_y_shape=0):
        f_spep = self.spep.function(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_shape = self.cartShapelets.function(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_spep + f_shape

    def derivatives(self, x, y, phi_E, gamma, q, phi_G, coeffs, beta, center_x=0, center_y=0, center_x_shape=0, center_y_shape=0):
        f_x_spep, f_y_spep = self.spep.derivatives(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_x_shape, f_y_shape = self.cartShapelets.derivatives(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_x_spep + f_x_shape, f_y_spep + f_y_shape

    def hessian(self, x, y, phi_E,gamma,q,phi_G, coeffs, beta, center_x=0, center_y=0, center_x_shape=0, center_y_shape=0):
        f_xx_spep, f_yy_spep, f_xy_spep = self.spep.hessian(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_xx_shape, f_yy_shape, f_xy_shape = self.cartShapelets.hessian(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_xx_spep + f_xx_shape, f_yy_spep + f_yy_shape, f_xy_spep + f_xy_shape

    def all(self, x, y, phi_E,gamma,q,phi_G, coeffs, beta, center_x=0, center_y=0, center_x_shape=0, center_y_shape=0):
        f_spep, f_x_spep, f_y_spep, f_xx_spep, f_yy_spep, f_xy_spep = self.spep.all(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_shape, f_x_shape, f_y_shape, f_xx_shape, f_yy_shape, f_xy_shape = self.cartShapelets.all(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_spep + f_shape, f_x_spep + f_x_shape, f_y_spep + f_y_shape, f_xx_spep + f_xx_shape, f_yy_spep + f_yy_shape, f_xy_spep + f_xy_shape



class SPEP_SPP_Shapelets(object):
    """
    combination of polar shapelets and SPEP profile
    """
    def __init__(self, type='SPEP'):
        from astrofunc.LensingProfiles.spep_spp import SPEP_SPP
        self.spep_spp = SPEP_SPP(type)
        from astrofunc.LensingProfiles.shapelet_pot_2 import CartShapelets
        self.cartShapelets = CartShapelets()

    def function(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coeffs, beta, center_x_shape=0, center_y_shape=0):
        f_spep_spp = self.spep_spp.function(x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp)
        f_shape = self.cartShapelets.function(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_spep_spp + f_shape

    def derivatives(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coeffs, beta, center_x_shape=0, center_y_shape=0):
        f_x_spep_spp, f_y_spep_spp = self.spep_spp.derivatives(x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp)
        f_x_shape, f_y_shape = self.cartShapelets.derivatives(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_x_spep_spp + f_x_shape, f_y_spep_spp + f_y_shape

    def hessian(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coeffs, beta, center_x_shape=0, center_y_shape=0):
        f_xx_spep_spp, f_yy_spep_spp, f_xy_spep_spp = self.spep_spp.hessian(x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp)
        f_xx_shape, f_yy_shape, f_xy_shape = self.cartShapelets.hessian(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_xx_spep_spp + f_xx_shape, f_yy_spep_spp + f_yy_shape, f_xy_spep_spp + f_xy_shape

    def all(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coeffs, beta, center_x_shape=0, center_y_shape=0):
        f_spep_spp, f_x_spep_spp, f_y_spep_spp, f_xx_spep_spp, f_yy_spep_spp, f_xy_spep_spp = self.spep_spp.all(x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp)
        f_shape, f_x_shape, f_y_shape, f_xx_shape, f_yy_shape, f_xy_shape = self.cartShapelets.all(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_spep_spp + f_shape, f_x_spep_spp + f_x_shape, f_y_spep_spp + f_y_shape, f_xx_spep_spp + f_xx_shape, f_yy_spep_spp + f_yy_shape, f_xy_spep_spp + f_xy_shape

