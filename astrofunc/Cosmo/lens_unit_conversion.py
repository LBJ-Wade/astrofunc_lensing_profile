from astropy.cosmology import default_cosmology
import astrofunc.constants as constants
from astrofunc.Cosmo.nfw_param import NFWParam

import numpy as np




class LensUnits(object):
    """
    class to convert various units (angles to mass etc) for a specific lens system and cosmology
    """
    def __init__(self, z_lens, z_source, cosmo=None):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param cosmo: astropy.cosmology instance
        """
        self._z_d = z_lens
        self._z_s = z_source

        if cosmo is None:
            self.cosmo = default_cosmology.get()
        else:
            self.cosmo = cosmo
        self.nfw_param = NFWParam()

    def a_z(self, z):
        """
        returns scale factor (a_0 = 1) for given redshift
        """
        return 1./(1+z)

    @property
    def h(self):
        return self.cosmo.H(0).value / 100.

    @property
    def D_d(self):
        if not hasattr(self, '_D_d'):
            self._D_d = self.cosmo.angular_diameter_distance(self._z_d)
        return self._D_d.value

    @property
    def D_s(self):
        if not hasattr(self, '_D_s'):
            self._D_s = self.cosmo.angular_diameter_distance(self._z_s)
        return self._D_s.value

    @property
    def D_ds(self):
        if not hasattr(self, '_D_ds'):
            a_s = self.a_z(self._z_s)
            a_d = self.a_z(self._z_d)
            self._D_ds = (self.D_s/a_s - self.D_d/a_d)*a_s
        return self._D_ds

    @property
    def epsilon_crit(self):
        """
        returns the critical projected mass density in units of M_sun/Mpc^2 (physical units)
        """
        if not hasattr(self,'_Epsilon_Crit'):
            const_SI = constants.c**2/(4*np.pi * constants.G)  #c^2/(4*pi*G) in units of [kg/m]
            conversion = constants.Mpc/constants.M_sun  # converts [kg/m] to [M_sun/Mpc]
            const = const_SI*conversion   #c^2/(4*pi*G) in units of [M_sun/Mpc]
            self._Epsilon_Crit = self.D_s/(self.D_d*self.D_ds) * const #[M_sun/Mpc^2]
        return self._Epsilon_Crit

    def nfw_angle2physical(self, Rs_angle, theta_Rs):
        """
        converts the angular parameters into the physical ones for an NFW profile
        :param theta_Rs: observed bending angle at the scale radius in units of arcsec
        :param Rs: scale radius in units of arcsec
        :return: M200, r200, Rs_physical, c
        """
        Rs = Rs_angle * constants.arcsec * self.D_d
        theta_scaled = theta_Rs * self.epsilon_crit * self.D_d * constants.arcsec
        rho0 = theta_scaled / (4 * Rs ** 2 * (1 + np.log(1. / 2.)))
        rho0_com = rho0 * self.h**2 * self.a_z(self._z_d)**3
        c = self.nfw_param.c_rho0(rho0_com)
        r200 = c * Rs
        M200 = self.nfw_param.M_r200(r200 / self.h / self.a_z(self._z_d)) / self.h
        return rho0, Rs, c, r200, M200

    def nfw_physical2angle(self, M, c):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """
        rho0, Rs, r200 = self.nfwParam_physical(M, c)
        Rs_angle = Rs / self.D_d / constants.arcsec  # Rs in arcsec
        theta_Rs = rho0 * (4 * Rs ** 2 * (1 + np.log(1. / 2.)))
        return Rs_angle,  theta_Rs / self.epsilon_crit / self.D_d / constants.arcsec

    def sis_angle2physical(self, theta_E):
        """
        converts the lensing Einstein radius into a physical velocity dispersion
        :param theta_E: Einstein radius (in arcsec)
        :return: velocity dispersion in units (km/s)
        """
        v_sigma_c2 = theta_E * constants.arcsec / (4*np.pi) * self.D_s / self.D_ds
        return np.sqrt(v_sigma_c2)*constants.c / 1000

    def sis_physical2angle(self, v_sigma):
        """
        converts the velocity dispersion into an Einstein radius for a SIS profile
        :param v_sigma: velocity dispersion (km/s)
        :return: theta_E (arcsec)
        """
        theta_E = 4 * np.pi * (v_sigma * 1000./constants.c)**2 * self.D_ds / self.D_s / constants.arcsec
        return theta_E

    def nfwParam_physical(self, M, c):
        """
        returns the NFW parameters in physical units
        :param M: physical mass in M_sun
        :param c: concentration
        :return:
        """
        r200 = self.nfw_param.r200_M(M * self.h) * self.h * self.a_z(self._z_d)  # physical radius r200
        rho0 = self.nfw_param.rho0_c(c) / self.h**2 / self.a_z(self._z_d)**3 # physical density in M_sun/Mpc**3
        Rs = r200/c
        return rho0, Rs, r200