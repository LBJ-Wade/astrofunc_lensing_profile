import numpy as np
import numpy.testing as npt
import pytest

from astrofunc.Cosmo.lens_unit_conversion import LensUnits
from astropy.cosmology import WMAP9


class TestLensUnits(object):

    def setup(self):
        z_d = 0.5
        z_s = 2
        self.lensUnits = LensUnits(z_lens=z_d, z_source=z_s, cosmo=WMAP9)

    def test_D_d(self):
        D_d = self.lensUnits.D_d
        assert D_d == 1277.37961573603

    def test_D_s(self):
        D_s = self.lensUnits.D_s
        assert D_s == 1763.9101459409605

    def test_D_ds(self):
        D_ds = self.lensUnits.D_ds
        assert D_ds == 1125.2203380729454

    def test_epsilon_crit(self):
        Epsilon_crit = self.lensUnits.epsilon_crit
        assert Epsilon_crit == 2040180453480701.5

if __name__ == '__main__':
    pytest.main()