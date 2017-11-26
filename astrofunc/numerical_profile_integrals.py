import copy
import scipy.integrate as integrate
import numpy as np
import scipy.signal as scp
import util


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


class ConvergenceIntegrals(object):
    """
    class to compute lensing potentials and deflection angles provided a convergence map
    """
    def potential_from_kappa(self, kappa, x_grid, y_grid, deltaPix):
        """

        :param kappa:
        :param x_coords:
        :param y_coords:
        :return:
        """
        kernel = self._potential_kernel(x_grid, y_grid)
        f_ = scp.fftconvolve(kernel, util.array2image(kappa), mode='same') / np.pi * deltaPix**2
        return f_

    def deflection_from_kappa(self, kappa, x_grid, y_grid, deltaPix):
        """

        :param kappa:
        :param x_grid:
        :param y_grid:
        :param deltaPix:
        :return:
        """
        kernel_x, kernel_y = self._deflection_kernel(x_grid, y_grid)
        f_x = scp.fftconvolve(kernel_x, util.array2image(kappa), mode='same') / np.pi * deltaPix**2
        f_y = scp.fftconvolve(kernel_y, util.array2image(kappa), mode='same') / np.pi * deltaPix ** 2
        return f_x, f_y

    def _potential_kernel(self, x_grid, y_grid):
        """

        :param numPix:
        :param deltaPix:
        :return:
        """
        x_mean = np.mean(x_grid)
        y_mean = np.mean(y_grid)
        r2 = (x_grid - x_mean)**2 + (y_grid - y_mean)**2
        r2_max = np.max(r2)
        lnr = np.log(r2/r2_max) / 2.
        lnr[r2 == 0] = 0
        kernel = util.array2image(lnr)
        return kernel

    def _deflection_kernel(self, x_grid, y_grid):
        """

        :param numPix:
        :param deltaPix:
        :return:
        """
        x_mean = np.mean(x_grid)
        y_mean = np.mean(y_grid)
        r2 = (x_grid - x_mean)**2 + (y_grid - y_mean)**2
        l0 = np.where(r2 == 0)

        kernel_x = util.array2image(x_grid / r2)
        kernel_y = util.array2image(y_grid / r2)
        kernel_x[l0] = 0
        kernel_y[l0] = 0
        return kernel_x, kernel_y
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

"""