__author__ = 'sibirrer'

class Ellipse(object):
    """
    class for choosing an elliptical lens model (SPEP or SPEMD)
    """
    def __init__(self, type='SPEP'):
        if type == 'SPEP':
            from astrofunc.LensingProfiles.spep import SPEP
            self.model = SPEP()
        if type == 'SPEMD':
            from astrofunc.LensingProfiles.spemd import SPEMD
            self.model = SPEMD()

    def function(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
        return self.model.function(x, y, theta_E, gamma, q, phi_G, center_x, center_y)

    def derivatives(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
        return self.model.derivatives(x, y, theta_E, gamma, q, phi_G, center_x, center_y)

    def hessian(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
        return self.model.hessian(x, y, theta_E, gamma, q, phi_G, center_x, center_y)

    def all(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
        return self.model.all(x, y, theta_E, gamma, q, phi_G, center_x, center_y)
