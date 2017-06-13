__author__ = 'sibirrer'

#this file contains a class to compute the Navaro-Frank-White function in mass/kappa space
#the potential therefore is its integral

import numpy as np
from astrofunc.LensingProfiles.nfw import NFW

class NFW_ELLIPSE(object):
    """
    this class contains functions concerning the NFW profile

    relation are: R_200 = c * Rs
    """
    def __init__(self):
        self.nfw = NFW()

    def function(self, x, y, Rs, theta_Rs, q, phi_G, center_x=0, center_y=0):
        """
        returns double integral of NFW profile
        """

        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1 = (cos_phi*x_shift+sin_phi*y_shift)*np.sqrt(1 - q)
        xt2 = (-sin_phi*x_shift+cos_phi*y_shift)*np.sqrt(1 + q)
        R_ = np.sqrt(xt1**2 + xt2**2)
        rho0_input = self.nfw._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0001:
            Rs = 0.0001
        f_ = self.nfw.nfwPot(R_, Rs, rho0_input)
        return f_

    def derivatives(self, x, y, Rs, theta_Rs, q, phi_G, center_x=0, center_y=0, angle=False):
        """
        returns df/dx and df/dy of the function (integral of NFW)
        """
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1 = (cos_phi*x_shift+sin_phi*y_shift)*np.sqrt(1 - q)
        xt2 = (-sin_phi*x_shift+cos_phi*y_shift)*np.sqrt(1 + q)
        R_ = np.sqrt(xt1**2 + xt2**2)
        rho0_input = self.nfw._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0001:
            Rs = 0.0001
        f_x_prim, f_y_prim = self.nfw.nfwAlpha(R_, Rs, rho0_input, xt1, xt2)
        f_x_prim *= np.sqrt(1 - q)
        f_y_prim *= np.sqrt(1 + q)
        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, Rs, theta_Rs, q, phi_G, center_x=0, center_y=0, angle=False):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        #TODO: not exactly correct: see Bolse & Kneib 2002
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1 = (cos_phi*x_shift+sin_phi*y_shift)*np.sqrt(1 - q)
        xt2 = (-sin_phi*x_shift+cos_phi*y_shift)*np.sqrt(1 + q)
        R_ = np.sqrt(xt1**2 + xt2**2)
        rho0_input = self.nfw._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0001:
            Rs = 0.0001
        kappa = self.nfw.nfw2D(R_, Rs, rho0_input)
        gamma1_value, gamma2_value = self.nfw.nfwGamma(R_, Rs, rho0_input, xt1, xt2)

        gamma1 = np.cos(2*phi_G)*gamma1_value-np.sin(2*phi_G)*gamma2_value
        gamma2 = +np.sin(2*phi_G)*gamma1_value+np.cos(2*phi_G)*gamma2_value
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def all(self, x, y, Rs, theta_Rs, q, phi_G, center_x=0, center_y=0, angle=False):
        """
        returns f,f_x,f_y,f_xx, f_yy, f_xy
        """
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1 = (cos_phi*x_shift+sin_phi*y_shift)*np.sqrt(1 - q)
        xt2 = (-sin_phi*x_shift+cos_phi*y_shift)*np.sqrt(1 + q)
        R_ = np.sqrt(xt1**2 + xt2**2)
        rho0_input = self.nfw._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0001:
            Rs = 0.0001

        f_ = self.nfw.nfwPot(R_, Rs, rho0_input)
        f_x_prim, f_y_prim = self.nfw.nfwAlpha(R_, Rs, rho0_input, xt1, xt2)
        f_x_prim *= np.sqrt(1 - q)
        f_y_prim *= np.sqrt(1 + q)
        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        kappa = self.nfw.nfw2D(R_, Rs, rho0_input)
        gamma1_value, gamma2_value = self.nfw.nfwGamma(R_, Rs, rho0_input, xt1, xt2)

        gamma1 = np.cos(2*phi_G)*gamma1_value-np.sin(2*phi_G)*gamma2_value
        gamma2 = +np.sin(2*phi_G)*gamma1_value+np.cos(2*phi_G)*gamma2_value
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_, f_x, f_y, f_xx, f_yy, f_xy
