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

    def test_hessian(self):
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

if __name__ == '__main__':
    pytest.main()