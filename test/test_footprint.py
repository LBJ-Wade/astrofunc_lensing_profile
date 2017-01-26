__author__ = 'sibirrer'



"""
Tests for `CheckFootprint` module.
"""

from astrofunc.Footprint.footprint import CheckFootprint

import pytest
import os
import numpy as np



class TestCheckFootprint(object):

    def setup(self):
        self.checkFootprint = CheckFootprint()
        filename = 'DESround13.txt'
        filename = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'PackageData',filename))
        self.raFoot,self.decFoot = self.checkFootprint.get_survey_data(filename)
        np.random.seed(seed=42)

    def test_select_survey(self):
        filename_test = self.checkFootprint.select_survey('DES')
        print(filename_test)
        assert os.path.isfile(filename_test)

        #TODO do the same with a non-existing survey name

    def test_get_survey_data(self):
        assert self.raFoot[0] == 23.00000
        assert self.decFoot[0] == -7.00000

    def test_check_footprint(self):
        assert self.checkFootprint.check_footprint(0.,0.,'DES') == True
        assert self.checkFootprint.check_footprint(100.,100.,'DES') == False

    def test_area_footprint(self):
        DES_area = self.checkFootprint.area_footprint(self.raFoot,self.decFoot,1000)
        assert DES_area > 4700 and DES_area < 5529


if __name__ == '__main__':
    pytest.main()