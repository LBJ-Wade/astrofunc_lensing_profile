__author__ = 'sibirrer'


from astrofunc.LensingProfiles.sersic import Sersic

import numpy as np
import pytest
import numpy.testing as npt


class TestSersic(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.sersic = Sersic()


    def test_function(self):
        x = 1
        y = 2
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        values = self.sersic.function(x, y, n_sersic, r_eff, k_eff)
        assert values == 1.0272982586319199

        x = np.array([0])
        y = np.array([0])
        values = self.sersic.function(x, y, n_sersic, r_eff, k_eff)
        npt.assert_almost_equal(values[0], 0., decimal=10)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.sersic.function(x, y, n_sersic, r_eff, k_eff)
        assert values[0] == 1.0272982586319199
        assert values[1] == 1.3318743892966658
        assert values[2] == 1.584299393114988

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        f_x, f_y = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff)
        assert f_x[0] == 0.18009724438623856
        assert f_y[0] == 0.36019448877247712
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff)
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff)
        assert values[0][0] == 0.18009724438623856
        assert values[1][0] == 0.36019448877247712
        assert values[0][1] == 0.26531952269062181
        assert values[1][1] == 0.088439840896873942

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        f_xx, f_yy,f_xy = self.sersic.hessian(x, y, n_sersic, r_eff, k_eff)
        assert f_xx[0] == 0.19307780179705503
        assert f_yy[0] == 0.23201947402950435
        assert f_xy[0] == 0.02596111482163288
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.sersic.hessian(x, y, n_sersic, r_eff, k_eff)
        assert values[0][0] == 0.19307780179705503
        assert values[1][0] == 0.23201947402950435
        assert values[2][0] == 0.02596111482163288
        assert values[0][1] == 0.10911239296600714
        assert values[1][1] == 0.090736791126777641
        assert values[2][1] == 0.006890850689711063

    def test_all(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.sersic.all(x, y, n_sersic, r_eff, k_eff)
        assert f_[0] == 1.0272982586319199
        assert f_x[0] == 0.18009724438623856
        assert f_y[0] == 0.36019448877247712
        assert f_xx[0] == 0.19307780179705503
        assert f_yy[0] == 0.23201947402950435
        assert f_xy[0] == 0.02596111482163288

if __name__ == '__main__':
    pytest.main()