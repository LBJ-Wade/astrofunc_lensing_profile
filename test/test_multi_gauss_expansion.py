__author__ = 'sibirrer'



import astrofunc.multi_gauss_expansion as mge

import numpy as np
import numpy.testing as npt
from astrofunc.LightProfiles.sersic import Sersic
from astrofunc.LightProfiles.gaussian import MultiGaussian
import pytest

class TestMGE(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.sersic = Sersic()
        self.multiGaussian = MultiGaussian()

    def test_mge_1d_sersic(self):
        n_comp = 30
        r_sersic = 1.
        n_sersic = 3.7
        I0_sersic = 1.
        rs = np.logspace(-2., 1., 50) * r_sersic
        ss = self.sersic.function(rs, np.zeros_like(rs), I0_sersic=I0_sersic, n_sersic=n_sersic, R_sersic=r_sersic)

        amplitudes, sigmas, norm = mge.mge_1d(rs, ss, N=n_comp)
        ss_mge = self.multiGaussian.function(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        #print((ss - ss_mge)/ss)
        for i in range(10, len(ss)-10):
            #print(rs[i])
            npt.assert_almost_equal((ss_mge[i]-ss[i])/ss[i], 0, decimal=1)

    def test_mge_sersic_radius(self):
        n_comp = 30
        r_sersic = .5
        n_sersic = 3.7
        I0_sersic = 1.
        rs = np.logspace(-2., 1., 50) * r_sersic
        ss = self.sersic.function(rs, np.zeros_like(rs), I0_sersic=I0_sersic, n_sersic=n_sersic, R_sersic=r_sersic)

        amplitudes, sigmas, norm = mge.mge_1d(rs, ss, N=n_comp)
        ss_mge = self.multiGaussian.function(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        print((ss - ss_mge)/(ss+ ss_mge))
        for i in range(10, len(ss)-10):
            #print(rs[i])
            npt.assert_almost_equal((ss_mge[i]-ss[i])/(ss[i]), 0, decimal=1)

    def test_mge_sersic_n_sersic(self):
        n_comp = 20
        r_sersic = 1.5
        n_sersic = .5
        I0_sersic = 1.
        rs = np.logspace(-2., 1., 50) * r_sersic
        ss = self.sersic.function(rs, np.zeros_like(rs), I0_sersic=I0_sersic, n_sersic=n_sersic, R_sersic=r_sersic)

        amplitudes, sigmas, norm = mge.mge_1d(rs, ss, N=n_comp)
        ss_mge = self.multiGaussian.function(rs, np.zeros_like(rs), amp=amplitudes, sigma=sigmas)
        for i in range(10, len(ss)-10):
            npt.assert_almost_equal((ss_mge[i]-ss[i])/(ss[i]+ss_mge[i]), 0, decimal=1)




if __name__ == '__main__':
    pytest.main()