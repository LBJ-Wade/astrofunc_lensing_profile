__author__ = 'sibirrer'
import pytest
import numpy as np

import astrofunc.util as util
from astrofunc.LensingProfiles.sis import SIS
from astrofunc.LensingProfiles.interpol import Interpol_func

class TestInterpol(object):

    def test_do_interpol(self):
        numPix = 100
        deltaPix = 0.1
        x_grid_interp, y_grid_interp = util.make_grid(numPix,deltaPix)
        sis = SIS()
        kwargs_SIS = {'theta_E_sis':1., 'center_x_sis':0.5, 'center_y_sis':-0.5}
        f_sis, f_x_sis, f_y_sis, f_xx_sis, f_yy_sis, f_xy_sis = sis.all(x_grid_interp, y_grid_interp, **kwargs_SIS)
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        interp_func = Interpol_func()
        interp_func.do_interp(x_axes, y_axes, util.array2image(f_sis), util.array2image(f_x_sis), util.array2image(f_y_sis), util.array2image(f_xx_sis), util.array2image(f_yy_sis), util.array2image(f_xy_sis))

        # test derivatives
        assert interp_func.derivatives(1,0) == sis.derivatives(1,0, **kwargs_SIS)
        alpha1_interp, alpha2_interp = interp_func.derivatives(np.array([0,1,0,1]), np.array([1,1,2,2]))
        alpha1_true, alpha2_true = sis.derivatives(np.array([0,1,0,1]),np.array([1,1,2,2]), **kwargs_SIS)
        assert alpha1_interp[0] == alpha1_true[0]
        assert alpha1_interp[1] == alpha1_true[1]
        # test hessian
        assert interp_func.hessian(1,0) == sis.hessian(1,0, **kwargs_SIS)
        f_xx_interp, f_yy_interp, f_xy_interp = interp_func.hessian(np.array([0,1,0,1]), np.array([1,1,2,2]))
        f_xx_true, f_yy_true, f_xy_true = sis.hessian(np.array([0,1,0,1]),np.array([1,1,2,2]), **kwargs_SIS)
        assert f_xx_interp[0] == f_xx_true[0]
        assert f_xx_interp[1] == f_xx_true[1]
        assert f_xy_interp[0] == f_xy_true[0]
        assert f_xy_interp[1] == f_xy_true[1]
        # test all
        assert interp_func.all(1,0) == sis.all(1,0, **kwargs_SIS)
        f_interp, f_x_interp, f_y_interp, f_xx_interp, f_yy_interp, f_xy_interp = interp_func.all(np.array([0,1,0,1]), np.array([1,1,2,2]))
        f_true, f_x_true, f_y_true, f_xx_true, f_yy_true, f_xy_true = sis.all(np.array([0,1,0,1]),np.array([1,1,2,2]), **kwargs_SIS)
        assert f_xx_interp[0] == f_xx_true[0]
        assert f_xx_interp[1] == f_xx_true[1]
        assert f_xy_interp[0] == f_xy_true[0]
        assert f_xy_interp[1] == f_xy_true[1]


if __name__ == '__main__':
    pytest.main("-k TestSourceModel")