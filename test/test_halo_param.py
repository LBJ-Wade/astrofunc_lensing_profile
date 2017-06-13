from astrofunc.Cosmo.halo_param import HaloParam
import numpy as np
import numpy.testing as npt
import pytest


class TestHaloParam(object):

    def setup(self):
        self.haloParam = HaloParam()


    def test_profile_main(self):
        M = 10**13
        z = 0.5
        r200, rho0, c, Rs = self.haloParam.profileMain(M, z)
        print(r200, np.log10(rho0), c, Rs)
        assert c == 3.9209051266072716

    def test_mass2angle(self):
        M = 10**13
        z = 0.5
        #alpha (physical units)/ self.lensProp.sigma_crit / self.lensProp.dist_OL / constants.arcsec
        assert 0 == 0


if __name__ == '__main__':
    pytest.main()