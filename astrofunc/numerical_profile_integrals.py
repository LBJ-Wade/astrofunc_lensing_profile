import copy
import scipy.integrate as integrate
import numpy as np
import scipy.signal as scp


class ProfileIntegrals(object):
    """
    class to perform integrals of spherical profiles to compute:
    - projected densities
    - enclosed densities
    - projected enclosed densities
    """
    def __init__(self, profile_class):
        self._profile = profile_class

    def mass_enclosed_3d(self, r, kwargs_profile):
        """
        computes the mass enclosed within a sphere of radius r
        :param r:
        :param kwargs_profile:
        :return:
        """
        kwargs = copy.deepcopy(kwargs_profile)
        try:
            del kwargs['center_x']
            del kwargs['center_y']
        except:
            pass
        # integral of self._profile.density(x)* 4*np.pi * x^2 *dx, 0,r
        out = integrate.quad(lambda x: self._profile.density(x, **kwargs)*4*np.pi*x**2, 0, r)
        return out[0]

    def density_2d(self, r, kwargs_profile):
        """
        computes the projected density along the line-of-sight
        :param r:
        :param kwargs_profile:
        :return:
        """
        kwargs = copy.deepcopy(kwargs_profile)
        try:
            del kwargs['center_x']
            del kwargs['center_y']
        except:
            pass
        # integral of self._profile.density(np.sqrt(x^2+r^2))* dx, 0, infty
        out = integrate.quad(lambda x: 2*self._profile.density(np.sqrt(x**2+r**2), **kwargs), 0, 100)
        return out[0]

    def mass_enclosed_2d(self, r, kwargs_profile):
        """
        computes the mass enclosed the projected line-of-sight
        :param r:
        :param kwargs_profile:
        :return:
        """
        kwargs = copy.deepcopy(kwargs_profile)
        try:
            del kwargs['center_x']
            del kwargs['center_y']
        except:
            pass
        # integral of self.density_2d(x)* 2*np.pi * x *dx, 0, r
        out = integrate.quad(lambda x: self.density_2d(x, kwargs)*2*np.pi*x, 0, r)
        return out[0]

    def potential_from_kappa(self, kappa, x_coords, y_coords):
        """

        :param kappa:
        :param x_coords:
        :param y_coords:
        :return:
        """


        def alpha_def(kappa, n1, n2, x0, y0, extra):
            # Computes the deflection angle of a single photon at coordinates theta in the source plane and a lens
            # mass distribution kappa

            nk1, nk2 = np.shape(kappa)
            x0 += extra / 2.
            y0 += extra / 2.
            # Coordonnees de la grille de l'espace image
            [x, y] = np.where(np.zeros([nk1, nk2]) == 0)
            [xk, yk] = np.where(np.zeros([nk1, nk2]) == 0)

            xc = np.reshape((x) - x0, (nk1, nk2))
            yc = np.reshape((y) - y0, (nk1, nk2))
            xkc = np.reshape((xk) - (nk1 / 2.), (nk1, nk2))
            ykc = np.reshape((yk) - (nk2 / 2.), (nk1, nk2))

            r = (xc ** 2 + yc ** 2)
            rk = np.sqrt(xkc ** 2 + ykc ** 2)
            lx, ly = np.where(r == 0)
            tabx = np.reshape((xc) / r, (nk1, nk2))
            taby = np.reshape((yc) / r, (nk1, nk2))
            tabx[lx, ly] = 0
            taby[lx, ly] = 0

            l = 0

            kappa = kappa.astype(float)
            tabx = tabx.astype(float)

            #   kappa[rk>(nk1)/2.] = 0

            intex = scp.fftconvolve(tabx, (kappa), mode='same') / np.pi
            intey = scp.fftconvolve(taby, (kappa), mode='same') / np.pi