__author__ = 'sibirrer'


import numpy as np

class SPEP(object):
    """
    class for Softened power-law elliptical potential (SPEP)
    """

    def function(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
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
        if gamma < 1.4:
            gamma = 1.4
        if gamma > 2.9:
            gamma = 2.9
        if q < 0.3:
            q = 0.3
        theta_E *= q
        x_shift = x - center_x
        y_shift = y - center_y
        E = theta_E / (((3 - gamma) / 2.) ** (1. / (1 - gamma)) * np.sqrt(q))
        #E = phi_E
        eta = -gamma+3
        xt1 = np.cos(phi_G)*x_shift+np.sin(phi_G)*y_shift
        xt2 = -np.sin(phi_G)*x_shift+np.cos(phi_G)*y_shift
        p2 = xt1**2+xt2**2/q**2
        s2 = 0. # softening
        return 2 * E**2/eta**2 * ((p2 + s2)/E**2)**(eta/2)

    def derivatives(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):

        # # @hope.jit
        # def xy_prime(dx, dy, eta, a, E, xt1, xt2, q):
        #     fac = 1./eta*(a/(E*E))**(eta/2-1)*2
        #     dx[:] = fac*xt1
        #     dy[:] = fac*xt2/(q*q)
        if gamma < 1.4:
            gamma = 1.4
        if gamma > 2.9:
            gamma = 2.9
        if q < 0.3:
            q = 0.3
        phi_E_new = theta_E * q
        x_shift = x - center_x
        y_shift = y - center_y
        E = phi_E_new / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        # E = phi_E
        eta = -gamma+3
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1=cos_phi*x_shift+sin_phi*y_shift
        xt2=-sin_phi*x_shift+cos_phi*y_shift
        xt2difq2 = xt2/(q*q)
        P2=xt1*xt1+xt2*xt2difq2
        if isinstance(P2, int) or isinstance(P2, float):
            a = max(0.000001,P2)
        else:
            a=np.empty_like(P2)
            p2 = P2[P2 > 0]  #in the SIS regime
            a[P2 == 0] = 0.000001
            a[P2 > 0] = p2
        fac = 1./eta*(a/(E*E))**(eta/2-1)*2
        f_x_prim = fac*xt1
        f_y_prim = fac*xt2difq2

        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
        if gamma < 1.4:
            gamma = 1.4
        if gamma > 2.9:
            gamma = 2.9
        if q < 0.3:
            q = 0.3
        phi_E_new = theta_E * q
        x_shift = x - center_x
        y_shift = y - center_y
        E = phi_E_new / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        # E = phi_E
        eta = -gamma+3
        xt1 = np.cos(phi_G)*x_shift+np.sin(phi_G)*y_shift
        xt2 = -np.sin(phi_G)*x_shift+np.cos(phi_G)*y_shift
        P2 = xt1**2+xt2**2/q**2
        if isinstance(P2, int) or isinstance(P2, float):
            a = max(0.000001, P2)
        else:
            a=np.empty_like(P2)
            p2 = P2[P2>0]  #in the SIS regime
            a[P2==0] = 0.000001
            a[P2>0] = p2
        s2 = 0. # softening

        kappa=1./eta*(a/E**2)**(eta/2-1)*((eta-2)*(xt1**2+xt2**2/q**4)/a+(1+1/q**2))
        gamma1_value=1./eta*(a/E**2)**(eta/2-1)*(1-1/q**2+(eta/2-1)*(2*xt1**2-2*xt2**2/q**4)/a)
        gamma2_value=4*xt1*xt2/q**2*(1./2-1/eta)*(a/E**2)**(eta/2-2)/E**2

        gamma1 = np.cos(2*phi_G)*gamma1_value-np.sin(2*phi_G)*gamma2_value
        gamma2 = +np.sin(2*phi_G)*gamma1_value+np.cos(2*phi_G)*gamma2_value
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def all(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
        if gamma < 1.4:
            gamma = 1.4
        if gamma > 2.9:
            gamma = 2.9
        if q < 0.3:
            q = 0.3
        phi_E_new = theta_E * q
        x_shift = x - center_x
        y_shift = y - center_y
        E = phi_E_new / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        # E = phi_E
        eta = -gamma+3
        xt1 = np.cos(phi_G)*x_shift+np.sin(phi_G)*y_shift
        xt2 = -np.sin(phi_G)*x_shift+np.cos(phi_G)*y_shift
        P2 = xt1**2+xt2**2/q**2

        if isinstance(P2, int) or isinstance(P2, float):
            p2 = max(0.000001, P2)
        else:
            p2 = np.empty_like(P2)
            p = P2[P2 > 0]  # in the SIS regime
            p2[P2==0] = 0.000001
            p2[P2>0] = p
        s2 = 0.  # softening
        f_ = 2 * E**2/eta**2 * ((p2 + s2)/E**2)**(eta/2)

        f_x_prim = 1./eta*(p2/E**2)**(eta/2-1)*2*xt1
        f_y_prim = 1./eta*(p2/E**2)**(eta/2-1)*2*xt2/q**2
        f_x = np.cos(phi_G)*f_x_prim-np.sin(phi_G)*f_y_prim
        f_y = np.sin(phi_G)*f_x_prim+np.cos(phi_G)*f_y_prim

        kappa = 1./eta*(p2/E**2)**(eta/2-1)*((eta-2)*(xt1**2+xt2**2/q**4)/p2+(1+1/q**2))
        gamma1_value = 1./eta*(p2/E**2)**(eta/2-1)*(1-1/q**2+(eta/2-1)*(2*xt1**2-2*xt2**2/q**4)/p2)
        gamma2_value = 4*xt1*xt2/q**2*(1./2-1/eta)*(p2/E**2)**(eta/2-2)/E**2

        gamma1 = np.cos(2*phi_G)*gamma1_value-np.sin(2*phi_G)*gamma2_value
        gamma2 = +np.sin(2*phi_G)*gamma1_value+np.cos(2*phi_G)*gamma2_value
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_, f_x, f_y, f_xx, f_yy, f_xy