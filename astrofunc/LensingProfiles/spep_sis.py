__author__ = 'sibirrer'


class SPEP_SIS(object):
    """
    combination of SPEP and NFW profile
    """
    def __init__(self, type='SPEP'):
        from astrofunc.LensingProfiles.ellipse import Ellipse
        self.spep = Ellipse(type)
        from astrofunc.LensingProfiles.sis import SIS
        self.sis = SIS()

    def function(self, x, y, theta_E, gamma, q, phi_G, center_x, center_y, phi_E_sis, center_x_sis, center_y_sis):
        f_spep = self.spep.function(x, y, theta_E, gamma, q, phi_G, center_x, center_y)
        f_sis = self.sis.function(x, y, phi_E_sis, center_x_sis=center_x_sis, center_y_sis=center_y_sis)
        return f_spep + f_sis

    def derivatives(self, x, y, theta_E, gamma, q, phi_G, center_x, center_y, phi_E_sis, center_x_sis, center_y_sis):
        f_x_spep, f_y_spep = self.spep.derivatives(x, y, theta_E, gamma, q, phi_G, center_x, center_y)
        f_x_sis, f_y_sis = self.sis.derivatives(x, y, phi_E_sis, center_x_sis=center_x_sis, center_y_sis=center_y_sis)
        return f_x_spep + f_x_sis, f_y_spep + f_y_sis

    def hessian(self, x, y, theta_E, gamma, q, phi_G, center_x, center_y, phi_E_sis, center_x_sis, center_y_sis):
        f_xx_spep, f_yy_spep, f_xy_spep = self.spep.hessian(x, y, theta_E, gamma, q, phi_G, center_x, center_y)
        f_xx_sis, f_yy_sis, f_xy_sis = self.sis.hessian(x, y, phi_E_sis, center_x_sis=center_x_sis, center_y_sis=center_y_sis)
        return f_xx_spep + f_xx_sis, f_yy_spep + f_yy_sis, f_xy_spep + f_xy_sis

    def all(self, x, y, theta_E, gamma, q, phi_G, center_x, center_y, phi_E_sis, center_x_sis, center_y_sis):
        f_spep, f_x_spep, f_y_spep, f_xx_spep, f_yy_spep, f_xy_spep = self.spep.all(x, y, theta_E, gamma, q, phi_G, center_x, center_y)
        f_sis, f_x_sis, f_y_sis, f_xx_sis, f_yy_sis, f_xy_sis = self.sis.all(x, y, phi_E_sis, center_x_sis=center_x_sis, center_y_sis=center_y_sis)
        return f_spep + f_sis, f_x_spep + f_x_sis, f_y_spep + f_y_sis, f_xx_spep + f_xx_sis, f_yy_spep + f_yy_sis, f_xy_spep + f_xy_sis
