__author__ = 'sibirrer'


import numpy as np

class SPP(object):
    """
    class for Softened power-law elliptical potential (SPEP)
    """

    def function(self, x, y, theta_E, gamma, center_x=0, center_y=0):
        """
        :param x: set of x-coordinates
        :type x: array of size (n)
        :param theta_E: Einstein radius of lense
        :type theta_E: float.
        :param gamma: power law slope of mass profifle
        :type gamma: <2 float
        :param q: Axis ratio
        :type q: 0<q<1
        :param phi_G: position angel of SES
        :type q: 0<phi_G<pi/2
        :returns:  function
        :raises: AttributeError, KeyError
        """
        if gamma < 1.6:
            gamma = 1.6
        if gamma > 2.9:
            gamma = 2.9

        x_ = x - center_x
        y_ = y - center_y
        E = theta_E / ((3 - gamma) / 2.) ** (1. / (1 - gamma))
        # E = phi_E_spp
        eta= -gamma + 3

        p2 = x_**2+y_**2
        s2 = 0. # softening
        return 2 * E**2/eta**2 * ((p2 + s2)/E**2)**(eta/2)

    def derivatives(self, x, y, theta_E, gamma, center_x=0., center_y=0.):

        # # @hope.jit
        # def xy_prime(dx, dy, eta, a, E, xt1, xt2, q):
        #     fac = 1./eta*(a/(E*E))**(eta/2-1)*2
        #     dx[:] = fac*xt1
        #     dy[:] = fac*xt2/(q*q)
        if gamma < 1.6:
            gamma = 1.6
        if gamma > 2.9:
            gamma = 2.9

        xt1 = x - center_x
        xt2 = y - center_y
        E = theta_E / ((3 - gamma) / 2.) ** (1. / (1 - gamma))
        # E = phi_E_spp
        eta = -gamma + 3

        P2=xt1*xt1+xt2*xt2
        if isinstance(P2, int) or isinstance(P2, float):
            a = max(0.000001,P2)
        else:
            a=np.empty_like(P2)
            p2 = P2[P2 > 0]  #in the SIS regime
            a[P2==0] = 0.000001
            a[P2>0] = p2

        fac = 1./eta*(a/(E*E))**(eta/2-1)*2
        f_x = fac*xt1
        f_y = fac*xt2
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, center_x=0., center_y=0.):
        if gamma < 1.6:
            gamma = 1.6
        if gamma > 2.9:
            gamma = 2.9

        xt1 = x - center_x
        xt2 = y - center_y
        E = theta_E / ((3 - gamma) / 2.) ** (1. / (1 - gamma))
        # E = phi_E_spp
        eta = -gamma + 3

        P2 = xt1**2+xt2**2
        if isinstance(P2, int) or isinstance(P2, float):
            a = max(0.000001,P2)
        else:
            a=np.empty_like(P2)
            p2 = P2[P2>0]  #in the SIS regime
            a[P2==0] = 0.000001
            a[P2>0] = p2
        s2 = 0. # softening

        kappa = 1./eta*(a/E**2)**(eta/2-1)*((eta-2)*(xt1**2+xt2**2)/a+(1+1))
        gamma1 = 1./eta*(a/E**2)**(eta/2-1)*((eta/2-1)*(2*xt1**2-2*xt2**2)/a)
        gamma2 = 4*xt1*xt2*(1./2-1/eta)*(a/E**2)**(eta/2-2)/E**2

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def all(self, x, y, theta_E, gamma, center_x=0., center_y=0.):
        if gamma < 1.6:
            gamma = 1.6
        if gamma > 2.9:
            gamma = 2.9

        xt1 = x - center_x
        xt2 = y - center_y
        E = theta_E / ((3 - gamma) / 2.) ** (1. / (1 - gamma))
        # E = phi_E_spp
        eta = -gamma + 3
        P2 = xt1**2+xt2**2

        if isinstance(P2, int) or isinstance(P2, float):
            p2 = max(0.000001,P2)
        else:
            p2 = np.empty_like(P2)
            p = P2[P2>0]  #in the SIS regime
            p2[P2==0] = 0.000001
            p2[P2>0] = p
        s2 = 0. # softening
        f_ = 2 * E**2/eta**2 * ((p2 +s2)/E**2)**(eta/2)

        f_x = 1./eta*((p2)/E**2)**(eta/2-1)*2*xt1
        f_y = 1./eta*((p2)/E**2)**(eta/2-1)*2*xt2

        kappa = 1./eta*(p2/E**2)**(eta/2-1)*((eta-2)*(xt1**2+xt2**2)/p2+(1+1))
        gamma1 = 1./eta*(p2/E**2)**(eta/2-1)*((eta/2-1)*(2*xt1**2-2*xt2**2)/p2)
        gamma2 = 4*xt1*xt2*(1./2-1/eta)*(p2/E**2)**(eta/2-2)/E**2

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2

        return f_, f_x, f_y, f_xx, f_yy, f_xy