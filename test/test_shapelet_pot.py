__author__ = 'sibirrer'

from astrofunc.LensingProfiles.shapelet_pot import PolarShapelets
from astrofunc.LensingProfiles.shapelet_pot_2 import CartShapelets
import astrofunc.util as util

import numpy as np
import numpy.testing as npt
import pytest


class TestPolarShapelets(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.polarShapelets = PolarShapelets()
        self.cartShapelets = CartShapelets()


    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        beta = 1.
        coeffs = (1., 1.)
        values = self.cartShapelets.function(x, y, coeffs, beta)
        assert values[0] == 0.11180585426466891

        x = 1.
        y = 2.
        beta = 1.
        coeffs = (1., 1.)
        values = self.cartShapelets.function(x, y, coeffs, beta)
        assert values == 0.11180585426466891

        x = np.array([0])
        y = np.array([0])
        beta = 1.
        coeffs = (0, 1.)
        values = self.cartShapelets.function(x, y, coeffs, beta)
        assert values[0] == 0

        coeffs = (1, 1., 0, 0, 1, 1)
        values = self.cartShapelets.function(x, y, coeffs, beta)
        assert values[0] == 0.16524730314632363

        coeffs = (1, 1., 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        values = self.cartShapelets.function(x, y, coeffs, beta)
        assert values[0] == 0.16524730314632363

        coeffs = (0., 0., 0, 0, 0., 0., 0, 0, 0, 0, 0, 0, 0, 0, 0)
        values = self.cartShapelets.function(x, y, coeffs, beta)
        assert values[0] == 0

    def test_derivatives(self):
        """

        :return:
        """
        beta = 1.
        coeffs = [0,0,0,1.,0,0,0,0]
        kwargs_lens1 = {'coeffs': coeffs, 'beta': beta}

        x1 = 1.
        y1 = 2.
        f_x1, f_y1 = self.cartShapelets.derivatives(x1, y1, **kwargs_lens1)
        x2 = np.array([1.])
        y2 = np.array([2.])
        f_x2, f_y2 = self.cartShapelets.derivatives(x2, y2, **kwargs_lens1)
        assert f_x1 == f_x2[0]

        x3 = np.array([1., 0])
        y3 = np.array([2., 0])
        f_x3, f_y3 = self.cartShapelets.derivatives(x3, y3, **kwargs_lens1)
        assert f_x1 == f_x3[0]



    def test_all(self):
        kwargs_options = {'lens_type': 'SHAPELETS_CART'}
        #lensModel = LensModel(kwargs_options)
        numPix = 150
        deltaPix = 0.05
        subgrid_res = 2
        x, y = util.make_grid(numPix, deltaPix, subgrid_res)

        beta = 1.
        coeffs = [0,0,0,1.,0,0,0,0]
        kwargs_lens1 = {'coeffs': coeffs, 'beta': beta}
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.cartShapelets.all(x, y, **kwargs_lens1)
        c00 = (np.sqrt(2))/2.
        c20 = (-3-2-1)/2.
        c40 = np.sqrt(3)
        c22 = np.sqrt(2)/2.
        coeffs_true = [c00,0,0,c20,0,0,0,0,0,0,c40,0,c22]
        shapelets = self.cartShapelets._createShapelet(coeffs)

        kappa_true = self.cartShapelets.function(x, y, coeffs_true, beta)/2./beta**2
        kappa = util.array2image(f_xx + f_yy)/2
        kappa_true = util.array2image(kappa_true)


        npt.assert_almost_equal(kappa[0][0], kappa_true[0][0], 5)
        #assert kappa[0][0] == kappa_true[0][0]
        x = 1.
        y = 2.
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.cartShapelets.all(x, y, **kwargs_lens1)
        assert f_ == 0.032747176537766647
        assert f_x == 0.098241529613299933

        x = np.array(1.)
        y = np.array(2.)
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.cartShapelets.all(x, y, **kwargs_lens1)
        assert f_ == 0.032747176537766647
        assert f_x == 0.098241529613299933

if __name__ == '__main__':
    pytest.main()