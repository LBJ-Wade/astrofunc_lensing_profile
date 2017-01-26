__author__ = 'sibirrer'

#this file contains a class to compute the Navaro-Frank-White function in mass/kappa space
#the potential therefore is its integral

import numpy as np

class NFW(object):
    """
    this class contains functions concerning the NFW profile

    relation are: R_200 = c * Rs
    """

    def function(self, x, y, Rs, rho0, r200=100, center_x_nfw=0, center_y_nfw=0, angle=False):
        """
        returns double integral of NFW profile
        """
        if angle is True:
            rho0_input = self.alpha2rho0(phi_E=rho0, Rs=Rs)
        else:
            rho0_input = rho0
        if Rs < 0.0001:
            Rs = 0.0001
        x_ = x - center_x_nfw
        y_ = y - center_y_nfw
        R = np.sqrt(x_**2 + y_**2)
        f_ = self.nfwPot(R, Rs, rho0_input, r200)
        return f_

    def derivatives(self, x, y, Rs, rho0, r200=100, center_x_nfw=0, center_y_nfw=0, angle=False):
        """
        returns df/dx and df/dy of the function (integral of NFW)
        """
        if angle is True:
            rho0_input = self.alpha2rho0(phi_E=rho0, Rs=Rs)
        else:
            rho0_input = rho0
        if Rs < 0.0001:
            Rs = 0.0001
        x_ = x - center_x_nfw
        y_ = y - center_y_nfw
        R = np.sqrt(x_**2 + y_**2)
        f_x, f_y = self.nfwAlpha(R, Rs, rho0_input, r200, x_, y_)
        return f_x, f_y

    def hessian(self, x, y, Rs, rho0, r200=100, center_x_nfw=0, center_y_nfw=0, angle=False):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        if angle is True:
            rho0_input = self.alpha2rho0(phi_E=rho0, Rs=Rs)
        else:
            rho0_input = rho0
        if Rs < 0.0001:
            Rs = 0.0001
        x_ = x - center_x_nfw
        y_ = y - center_y_nfw
        R = np.sqrt(x_**2 + y_**2)
        kappa = self.nfw2D(R, Rs, rho0_input, r200)
        gamma1, gamma2 = self.nfwGamma(R, Rs, rho0_input, r200, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def all(self, x, y, Rs, rho0, r200=100, center_x_nfw=0, center_y_nfw=0, angle=False):
        """
        returns f,f_x,f_y,f_xx, f_yy, f_xy
        """
        if angle is True:
            rho0_input = self.alpha2rho0(phi_E=rho0, Rs=Rs)
        else:
            rho0_input = rho0
        if Rs < 0.0001:
            Rs = 0.0001
        x_ = x - center_x_nfw
        y_ = y - center_y_nfw
        R = np.sqrt(x_**2 + y_**2)
        f_ = self.nfwPot(R, Rs, rho0_input, r200)
        f_x, f_y = self.nfwAlpha(R, Rs, rho0_input, r200, x_, y_)
        kappa = self.nfw2D(R, Rs, rho0_input, r200)
        gamma1, gamma2 = self.nfwGamma(R, Rs, rho0_input, r200, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_, f_x, f_y, f_xx, f_yy, f_xy

    def nfw3D(self,R,Rs,rho0):
        """
        three dimenstional NFW profile

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :return: rho(R) density
        """
        return rho0/(R/Rs*(1+R/Rs)**2)

    def nfw2D(self,R,Rs,rho0,r200=1):
        """
        projected two dimenstional NFW profile (kappa*Sigma_crit)

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r200: radius of (sub)halo
        :type r200: float>0
        :return: Epsilon(R) projected density at radius R
        """
        x = R/Rs
        Fx = self.F(x)
        return 2*rho0*Rs*Fx

    def nfw2D_smoothed(self, R, Rs, rho0, r200, pixscale):
        """
        projected two dimenstional NFW profile with smoothing around the pixel scale
        this routine is ment to better compare outputs to N-body simulations (not ment ot do lensemodelling with it)

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r200: radius of (sub)halo
        :type r200: float>0
        :param pixscale: pixel scale (same units as R,Rs)
        :type pixscale: float>0
        :return: Epsilon(R) projected density at radius R
        """
        x = R/Rs
        d = pixscale/(2*Rs)
        a = np.empty_like(x)
        x_ = x[x > d]
        upper = x_+d
        lower = x_-d

        a[x > d] = 4*rho0*Rs**3*(self.g(upper)-self.g(lower))/(2*x_*Rs*pixscale)
        a[x < d] = 4*rho0*Rs**3*self.g(d)/((pixscale/2)**2)
        return a

    def nfwPot(self, R, Rs, rho0, r200=1):
        """
        lensing potential of NFW profile (*Sigma_crit*D_OL**2)

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r200: radius of (sub)halo
        :type r200: float>0
        :return: Epsilon(R) projected density at radius R
        """
        x=R/Rs
        hx=self.h(x)
        return 2*rho0*Rs**3*hx

    def nfwAlpha(self, R, Rs, rho0, r200, ax_x, ax_y):
        """
        deflection angel of NFW profile (*Sigma_crit*D_OL) along the projection to coordinate "axis"

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r200: radius of (sub)halo
        :type r200: float>0
        :param axis: projection to either x- or y-axis
        :type axis: same as R
        :return: Epsilon(R) projected density at radius R
        """
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, 0.00001)
        else:
            R[R==0] = 0.00001
        x = R/Rs
        gx = self.g(x)
        a = 4*rho0*Rs*R*gx/x**2/R
        return a*ax_x, a*ax_y

    def nfwGamma(self, R, Rs, rho0, r200, ax_x, ax_y):
        """
        shear gamma of NFW profile (*Sigma_crit) along the projection to coordinate "axis"

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r200: radius of (sub)halo
        :type r200: float>0
        :param axis: projection to either x- or y-axis
        :type axis: same as R
        :return: Epsilon(R) projected density at radius R
        """
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, 0.001)
        else:
            R[R==0] = 0.001
        x = R/Rs
        gx = self.g(x)
        Fx = self.F(x)
        a = 2*rho0*Rs*(2*gx/x**2 - Fx)#/x #2*rho0*Rs*(2*gx/x**2 - Fx)*axis/x
        return a*(ax_y**2-ax_x**2)/R**2, a*2*(ax_x*ax_y)/R**2

    def F(self, X):
        """
        analytic solution of the projection integral

        :param x: R/Rs
        :type x: float >0
        """
        if isinstance(X, int) or isinstance(X, float):
            if X < 1 and X > 0:
                a = 1/(X**2-1)*(1-2/np.sqrt(1-X**2)*np.arctanh(np.sqrt((1-X)/(1+X))))
            elif X == 1:
                a = 1./3
            elif X > 1:
                a = 1/(X**2-1)*(1-2/np.sqrt(X**2-1)*np.arctan(np.sqrt((X-1)/(1+X))))
            elif X == 0:
                c = 0.0001
                a = 1/(-1)*(1-2/np.sqrt(1)*np.arctanh(np.sqrt((1-c)/(1+c))))

        else:
            a=np.empty_like(X)
            x = X[X<1]
            a[X<1] = 1/(x**2-1)*(1-2/np.sqrt(1-x**2)*np.arctanh(np.sqrt((1-x)/(1+x))))

            x = X[X==1]
            a[X==1] = 1./3.

            x = X[X>1]
            a[X>1] = 1/(x**2-1)*(1-2/np.sqrt(x**2-1)*np.arctan(np.sqrt((x-1)/(1+x))))
            # a[X>y] = 0

            c = 0.001
            x = X[X==0]
            a[X==0] = 1/(-1)*(1-2/np.sqrt(1)*np.arctanh(np.sqrt((1-c)/(1+c))))
        return a

    def g(self, X):
        """
        analytic solution of integral for NFW profile to compute deflection angel and gamma

        :param x: R/Rs
        :type x: float >0
        """
        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                x = max(0.001, X)
                a = np.log(x/2.) + 1/np.sqrt(1-x**2)*np.arccosh(1./x)
            elif X == 1:
                a = 1 + np.log(1./2.)
            elif X > 1:
                a = np.log(X/2) + 1/np.sqrt(X**2-1)*np.arccos(1./X)

        else:
            a=np.empty_like(X)
            X[X==0] = 0.001
            x = X[X<1]

            a[X<1] = np.log(x/2.) + 1/np.sqrt(1-x**2)*np.arccosh(1./x)

            a[X==1] = 1 + np.log(1./2.)

            x = X[X>1]
            a[X>1] = np.log(x/2) + 1/np.sqrt(x**2-1)*np.arccos(1./x)

        return a

    def h(self, X):
        """
        analytic solution of integral for NFW profile to compute the potential

        :param x: R/Rs
        :type x: float >0
        """
        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                x = max(0.001, X)
                a = np.log(x/2.)**2 - np.arccosh(1./x)**2
            elif X >= 1:
                a = np.log(X/2.)**2 + np.arccos(1./X)**2
        else:
            a=np.empty_like(X)
            X[X==0] = 0.001
            x = X[(X<1) & (X>0)]
            a[(X<1) & (X>0)] = np.log(x/2.)**2 - np.arccosh(1./x)**2
            x = X[X >= 1]
            a[X >= 1] = np.log(x/2.)**2 + np.arccos(1./x)**2
        return a

    def alpha2rho0(self, phi_E, Rs):
        """
        convert angle at Rs into rho0
        """
        return phi_E/(4*Rs**2*(1+np.log(1./2.)))


class HaloParam(object):
    """
    class which contains a halo model parameters dependent on cosmology for NFW profile
    all distances are given in comoving coordinates
    """

    rhoc = 2.77536627e11  # critical density [h^2 M_sun Mpc^-3]

    def M200(self, Rs, rho0, c):
        """
        M(R_200) calculation for NFW profile

        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param c: consentration
        :type c: float [4,40]
        :return: M(R_200) density
        """
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c/(1+c))

    def r200_M(self, M):
        """
        computes the radius R_200 of a halo of mass M in comoving distances

        :param M: halo mass in M_sun/h
        :type M: float or numpy array
        :return: radius R_200 in comoving Mpc/h
        """
        return (3*M/(4*np.pi*self.rhoc*200))**(1./3.)

    def rho0_c(self, c):
        """
        computes density normalization as a functio of concentration parameter
        :return: density normalization in h^2/Mpc^3 (comoving)
        """
        return 200./3*self.rhoc*c**3/(np.log(1+c)-c/(1+c))

    def c_M_z(self, M, z):
        """
        fitting function of http://moriond.in2p3.fr/J08/proceedings/duffy.pdf for the mass and redshift dependence of the concentration parameter

        :param M: halo mass in M_sun/h
        :type M: float or numpy array
        :param z: redshift
        :type z: float >0
        :return: concentration parameter as float
        """
        # fitted parameter values
        A = 5.22
        B = -0.072
        C = -0.42
        M_pivot = 2*10**12
        return A*(M/M_pivot)**B*(1+z)**C

    def profileMain(self, M, z):
        """
        returns all needed parameter to draw the profile of the main halo
        """
        c = self.c_M_z(M,z)
        r200 = self.r200_M(M)
        rho0 = self.rho0_c(c)
        Rs = r200/c
        return r200,rho0,c,Rs