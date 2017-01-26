__author__ = 'sibirrer'


class SPEP_NFW(object):
    """
    combination of SPEP and NFW profile
    """
    def __init__(self, type='SPEP'):
        from lenstronomy.FunctionSet.ellipse import Ellipse
        self.spep = Ellipse(type)
        from lenstronomy.FunctionSet.nfw import NFW
        self.nfw = NFW()

    def function(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, Rs, rho0, r200, center_x_nfw, center_y_nfw):
        f_spep = self.spep.function(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_nfw = self.nfw.function(x, y, Rs, rho0, r200, center_x_nfw, center_y_nfw)
        return f_spep + f_nfw

    def derivatives(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, Rs, rho0, r200, center_x_nfw, center_y_nfw):
        f_x_spep, f_y_spep = self.spep.derivatives(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_x_nfw, f_y_nfw = self.nfw.derivatives(x, y, Rs, rho0, r200, center_x_nfw, center_y_nfw)
        return f_x_spep + f_x_nfw, f_y_spep + f_y_nfw

    def hessian(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, Rs, rho0, r200, center_x_nfw, center_y_nfw):
        f_xx_spep, f_yy_spep, f_xy_spep = self.spep.hessian(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_xx_nfw, f_yy_nfw, f_xy_nfw = self.nfw.hessian(x, y, Rs, rho0, r200, center_x_nfw, center_y_nfw)
        return f_xx_spep + f_xx_nfw, f_yy_spep + f_yy_nfw, f_xy_spep + f_xy_nfw

    def all(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, Rs, rho0, r200, center_x_nfw, center_y_nfw):
        f_spep, f_x_spep, f_y_spep, f_xx_spep, f_yy_spep, f_xy_spep = self.spep.all(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_nfw, f_x_nfw, f_y_nfw, f_xx_nfw, f_yy_nfw, f_xy_nfw = self.nfw.all(x, y, Rs, rho0, r200, center_x_nfw, center_y_nfw)
        return f_spep + f_nfw, f_x_spep + f_x_nfw, f_y_spep + f_y_nfw, f_xx_spep + f_xx_nfw, f_yy_spep + f_yy_nfw, f_xy_spep + f_xy_nfw