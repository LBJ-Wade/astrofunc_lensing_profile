import numpy as np


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
        M_pivot = 2.*10**12
        return A*(M/M_pivot)**B*(1+z)**C

    def profileMain(self, M, z):
        """
        returns all needed parameter (in comoving units modulo h) to draw the profile of the main halo
        """
        c = self.c_M_z(M,z)
        r200 = self.r200_M(M)
        rho0 = self.rho0_c(c)
        Rs = r200/c
        return r200, rho0, c, Rs