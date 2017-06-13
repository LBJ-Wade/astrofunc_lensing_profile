from astropy.cosmology import default_cosmology
import astrofunc.constants as constants

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

    def a_z(self, z):
        """
        returns scale factor (a_0 = 1) for given redshift
        """
        return 1./(1+z)

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