__author__ = 'sibirrer'

class SPEP_SPP_Dipole(object):
    """
    combination of SPEP and SPP profile
    """
    def __init__(self, type='SPEP'):
        from lenstronomy.FunctionSet.ellipse import Ellipse
        self.spep = Ellipse(type)
        from lenstronomy.FunctionSet.spp import SPP
        self.spp = SPP()
        from lenstronomy.FunctionSet.dipole import Dipole, Dipole_util
        self.dipole = Dipole()
        self.dipole_util = Dipole_util()

    def function(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole):
        f_spep = self.spep.function(x, y, phi_E,gamma, q, phi_G, center_x, center_y)
        f_spp = self.spp.function(x, y, phi_E_spp, gamma_spp, center_x_spp=center_x_spp, center_y_spp=center_y_spp)
        Fm = self.dipole_util.mass_ratio(phi_E, phi_E_spp)
        com_x, com_y = self.dipole_util.com(center_x, center_y, center_x_spp, center_y_spp, Fm)
        f_dipole = self.dipole.function(x, y, com_x=com_x, com_y=com_y, phi_dipole=phi_dipole, coupling=coupling)
        return f_spep + f_spp + f_dipole

    def derivatives(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole):
        f_x_spep, f_y_spep = self.spep.derivatives(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_x_spp, f_y_spp = self.spp.derivatives(x, y, phi_E_spp, gamma_spp, center_x_spp=center_x_spp, center_y_spp=center_y_spp)
        Fm = self.dipole_util.mass_ratio(phi_E, phi_E_spp)
        com_x, com_y = self.dipole_util.com(center_x, center_y, center_x_spp, center_y_spp, Fm)
        f_x_dip, f_y_dip = self.dipole.derivatives(x, y, com_x=com_x, com_y=com_y, phi_dipole=phi_dipole, coupling=coupling)
        return f_x_spep + f_x_spp + f_x_dip, f_y_spep + f_y_spp + f_y_dip

    def hessian(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole):
        f_xx_spep, f_yy_spep, f_xy_spep = self.spep.hessian(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_xx_spp, f_yy_spp, f_xy_spp = self.spp.hessian(x, y, phi_E_spp, gamma_spp, center_x_spp=center_x_spp, center_y_spp=center_y_spp)
        Fm = self.dipole_util.mass_ratio(phi_E, phi_E_spp)
        com_x, com_y = self.dipole_util.com(center_x, center_y, center_x_spp, center_y_spp, Fm)
        f_xx_dip, f_yy_dip, f_xy_dip = self.dipole.hessian(x, y, com_x=com_x, com_y=com_y, phi_dipole=phi_dipole, coupling=coupling)
        return f_xx_spep + f_xx_spp + f_xx_dip, f_yy_spep + f_yy_spp + f_yy_dip, f_xy_spep + f_xy_spp + f_xy_dip

    def all(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole):
        f_spep, f_x_spep, f_y_spep, f_xx_spep, f_yy_spep, f_xy_spep = self.spep.all(x, y, phi_E,gamma,q,phi_G, center_x, center_y)
        f_spp, f_x_spp, f_y_spp, f_xx_spp, f_yy_spp, f_xy_spp = self.spp.all(x, y, phi_E_spp, gamma_spp, center_x_spp=center_x_spp, center_y_spp=center_y_spp)
        Fm = self.dipole_util.mass_ratio(phi_E, phi_E_spp)
        com_x, com_y = self.dipole_util.com(center_x, center_y, center_x_spp, center_y_spp, Fm)
        f_dip, f_x_dip, f_y_dip, f_xx_dip, f_yy_dip, f_xy_dip = self.dipole.all(x, y, com_x=com_x, com_y=com_y, phi_dipole=phi_dipole, coupling=coupling)
        return f_spep + f_spp + f_dip, f_x_spep + f_x_spp + f_x_dip, f_y_spep + f_y_spp + f_y_dip, f_xx_spep + f_xx_spp + f_xx_dip, f_yy_spep + f_yy_spp + f_yy_dip, f_xy_spep + f_xy_spp + f_xy_dip


class SPEP_SPP_Dipole_Shapelets(object):
    """
    combination of cartesian shapelets with SPEP, SPP and dipole component
    """
    def __init__(self, type='SPEP'):
        self.spep_spp_dipole = SPEP_SPP_Dipole(type)
        from lenstronomy.FunctionSet.shapelet_pot_2 import CartShapelets
        self.cartShapelets = CartShapelets()

    def function(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole, coeffs, beta, center_x_shape=0, center_y_shape=0):
        f_spep_spp_dipole = self.spep_spp_dipole.function(x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole)
        f_shape = self.cartShapelets.function(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_spep_spp_dipole + f_shape

    def derivatives(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole, coeffs, beta, center_x_shape=0, center_y_shape=0):
        f_x_spep_spp, f_y_spep_spp = self.spep_spp_dipole.derivatives(x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole)
        f_x_shape, f_y_shape = self.cartShapelets.derivatives(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_x_spep_spp + f_x_shape, f_y_spep_spp + f_y_shape

    def hessian(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole, coeffs, beta, center_x_shape=0, center_y_shape=0):
        f_xx_spep_spp, f_yy_spep_spp, f_xy_spep_spp = self.spep_spp_dipole.hessian(x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole)
        f_xx_shape, f_yy_shape, f_xy_shape = self.cartShapelets.hessian(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_xx_spep_spp + f_xx_shape, f_yy_spep_spp + f_yy_shape, f_xy_spep_spp + f_xy_shape

    def all(self, x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole, coeffs, beta, center_x_shape=0, center_y_shape=0):
        f_spep_spp, f_x_spep_spp, f_y_spep_spp, f_xx_spep_spp, f_yy_spep_spp, f_xy_spep_spp = self.spep_spp_dipole.all(x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp, coupling, phi_dipole)
        f_shape, f_x_shape, f_y_shape, f_xx_shape, f_yy_shape, f_xy_shape = self.cartShapelets.all(x, y, coeffs, beta, center_x=center_x_shape, center_y=center_y_shape)
        return f_spep_spp + f_shape, f_x_spep_spp + f_x_shape, f_y_spep_spp + f_y_shape, f_xx_spep_spp + f_xx_shape, f_yy_spep_spp + f_yy_shape, f_xy_spep_spp + f_xy_shape