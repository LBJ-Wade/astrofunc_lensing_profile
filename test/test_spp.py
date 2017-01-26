__author__ = 'sibirrer'

from astrofunc.LensingProfiles.spep import SPEP
from astrofunc.LensingProfiles.spp import SPP

import numpy as np
import pytest

class TestSPEP(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.SPEP = SPEP()
        self.SPP = SPP()


    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 1
        phi_G = 0.
        E = phi_E / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        values_spep = self.SPEP.function(x, y, E, gamma,q,phi_G)
        values_spp = self.SPP.function(x, y, E, gamma)
        assert values_spep[0] == values_spp[0]
        x = np.array([0])
        y = np.array([0])
        values_spep = self.SPEP.function(x, y, E, gamma,q,phi_G)
        values_spp = self.SPP.function(x, y, E, gamma)
        assert values_spep[0] == values_spp[0]

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values_spep = self.SPEP.function(x, y, E, gamma,q,phi_G)
        values_spp = self.SPP.function(x, y, E, gamma)
        assert values_spep[0] == values_spp[0]
        assert values_spep[1] == values_spp[1]
        assert values_spep[2] == values_spp[2]

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 1
        phi_G = 0.
        E = phi_E / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        f_x_spep, f_y_spep = self.SPEP.derivatives(x, y, E, gamma,q,phi_G)
        f_x_spp, f_y_spp = self.SPP.derivatives(x, y, E, gamma)
        assert f_x_spep[0] == f_x_spp[0]
        assert f_y_spep[0] == f_y_spp[0]
        x = np.array([0])
        y = np.array([0])
        f_x_spep, f_y_spep = self.SPEP.derivatives(x, y, E, gamma,q,phi_G)
        f_x_spp, f_y_spp = self.SPP.derivatives(x, y, E, gamma)
        assert f_x_spep[0] == f_x_spp[0]
        assert f_y_spep[0] == f_y_spp[0]

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        f_x_spep, f_y_spep = self.SPEP.derivatives(x, y, E, gamma,q,phi_G)
        f_x_spp, f_y_spp = self.SPP.derivatives(x, y, E, gamma)
        assert f_x_spep[0] == f_x_spp[0]
        assert f_y_spep[0] == f_y_spp[0]
        assert f_x_spep[1] == f_x_spp[1]
        assert f_y_spep[1] == f_y_spp[1]
        assert f_x_spep[2] == f_x_spp[2]
        assert f_y_spep[2] == f_y_spp[2]

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 1.
        phi_G = 0.
        E = phi_E / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        f_xx, f_yy,f_xy = self.SPEP.hessian( x, y, E,gamma,q,phi_G)
        f_xx_spep, f_yy_spep, f_xy_spep = self.SPEP.hessian(x, y, E, gamma,q,phi_G)
        f_xx_spp, f_yy_spp, f_xy_spp = self.SPP.hessian(x, y, E, gamma)
        assert f_xx_spep[0] == f_xx_spp[0]
        assert f_yy_spep[0] == f_yy_spp[0]
        assert f_xy_spep[0] == f_xy_spp[0]
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        f_xx_spep, f_yy_spep, f_xy_spep = self.SPEP.hessian(x, y, E, gamma,q,phi_G)
        f_xx_spp, f_yy_spp, f_xy_spp = self.SPP.hessian(x, y, E, gamma)
        assert f_xx_spep[0] == f_xx_spp[0]
        assert f_yy_spep[0] == f_yy_spp[0]
        assert f_xy_spep[0] == f_xy_spp[0]
        assert f_xx_spep[1] == f_xx_spp[1]
        assert f_yy_spep[1] == f_yy_spp[1]
        assert f_xy_spep[1] == f_xy_spp[1]
        assert f_xx_spep[2] == f_xx_spp[2]
        assert f_yy_spep[2] == f_yy_spp[2]
        assert f_xy_spep[2] == f_xy_spp[2]

    def test_all(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 1
        phi_G = 0
        E = phi_E / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        f_spep, f_x_spep, f_y_spep, f_xx_spep, f_yy_spep, f_xy_spep = self.SPEP.all( x, y, E,gamma,q,phi_G)
        f_spp, f_x_spp, f_y_spp, f_xx_spp, f_yy_spp, f_xy_spp = self.SPP.all( x, y, E,gamma)
        assert f_spep[0] == f_spp[0]
        assert f_x_spep[0] == f_x_spp[0]
        assert f_y_spep[0] == f_y_spp[0]
        assert f_xx_spep[0] == f_xx_spp[0]
        assert f_yy_spep[0] == f_yy_spp[0]
        assert f_xy_spep[0] == f_xy_spp[0]

if __name__ == '__main__':
   pytest.main()