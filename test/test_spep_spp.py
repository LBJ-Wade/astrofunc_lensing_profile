__author__ = 'sibirrer'

from astrofunc.LensingProfiles.spep import SPEP
from astrofunc.LensingProfiles.spp import SPP
from astrofunc.LensingProfiles.spep_spp import SPEP_SPP

import numpy as np
import pytest

class TestSPEP_SPP(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.SPEP = SPEP()
        self.SPP = SPP()
        self.SPEP_SPP = SPEP_SPP()


    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp = 1., 1.9, 0.9, 1, 0, 0, 1, 1.8, 1,1
        values = self.SPEP.function(x, y, phi_E, gamma, q, phi_G, center_x, center_y) + self.SPP.function(x, y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp)
        values_new = self.SPEP_SPP.function(x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp)
        assert values[0] == values_new[0]


    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp = 1., 1.9, 0.9, 1, 0, 0, 1, 1.8, 1,1
        f_x1, f_y1 = self.SPEP.derivatives(x, y, phi_E, gamma, q, phi_G, center_x, center_y)
        f_x2, f_y2 = self.SPP.derivatives(x, y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp)
        f_x_new, f_y_new = self.SPEP_SPP.derivatives(x, y, phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp)
        f_x = f_x1 + f_x2
        f_y = f_y1 + f_y2
        assert f_x[0] == f_x_new[0]
        assert f_y[0] == f_y_new[0]


if __name__ == '__main__':
   pytest.main()