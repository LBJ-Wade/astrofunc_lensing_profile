__author__ = 'sibirrer'

class SPEP_SPP(object):
    """
    combination of SPEP and SPP profile
    """
    def __init__(self, type='SPEP'):
        from lenstronomy.FunctionSet.ellipse import Ellipse
        self.spep = Ellipse(type)
        from lenstronomy.FunctionSet.spp import SPP
        self.spp = SPP()

    def function(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp):
        f_spep = self.spep.function(x, y, phi_E,gamma, q, phi_G, center_x, center_y)
        f_spp = self.spp.function(x, y, phi_E_spp, gamma_spp, center_x_spp=center_x_spp, center_y_spp=center_y_spp)
        return f_spep + f_spp

    def derivatives(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp):
        f_x_spep, f_y_spep = self.spep.derivatives(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_x_spp, f_y_spp = self.spp.derivatives(x, y, phi_E_spp, gamma_spp, center_x_spp=center_x_spp, center_y_spp=center_y_spp)
        return f_x_spep + f_x_spp, f_y_spep + f_y_spp

    def hessian(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp):
        f_xx_spep, f_yy_spep, f_xy_spep = self.spep.hessian(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_xx_spp, f_yy_spp, f_xy_spp = self.spp.hessian(x, y, phi_E_spp, gamma_spp, center_x_spp=center_x_spp, center_y_spp=center_y_spp)
        return f_xx_spep + f_xx_spp, f_yy_spep + f_yy_spp, f_xy_spep + f_xy_spp

    def all(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp):
        f_spep, f_x_spep, f_y_spep, f_xx_spep, f_yy_spep, f_xy_spep = self.spep.all(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_spp, f_x_spp, f_y_spp, f_xx_spp, f_yy_spp, f_xy_spp = self.spp.all(x, y, phi_E_spp, gamma_spp, center_x_spp=center_x_spp, center_y_spp=center_y_spp)
        return f_spep + f_spp, f_x_spep + f_x_spp, f_y_spep + f_y_spp, f_xx_spep + f_xx_spp, f_yy_spep + f_yy_spp, f_xy_spep + f_xy_spp
