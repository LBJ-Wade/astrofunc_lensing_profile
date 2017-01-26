#
__author__ = "simonbirrer"

import numpy as np
import os


class FootprintUtil(object):
    """
    this class contains routines for asking the question whether a certain coordinate (ra/dec) is in a certain survey.
    Surveys are specified by a list of coordinates.
    Routines which compute the area of a given survey are also listed
    """

    def insideFootprint(self, ra, dec, raFoot, decFoot):
        if ra > 180: ra = ra - 360.
        poly = self.convertCoordsToPoly(raFoot, decFoot)
        return self.point_in_poly(ra, dec, poly)

    def area_footprint(self, raFoot, decFoot, n):
        random = self.uniform_sphere((-180, 180), (-90, 90), n)
        randRa = random[0];
        randDec = random[1]
        sum = 0
        for i in range(0, n):
            if self.insideFootprint(randRa[i], randDec[i], raFoot, decFoot):
                sum += 1
        frac = float(sum) / n
        sky = 41253
        area = round(frac * sky)
        return area

    def convertCoordsToPoly(self, ra, dec):
        poly = []
        for i in range(0, len(ra)):
            poly.append((ra[i], dec[i]))
        return poly

    def point_in_poly(self, x, y, poly):
        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def uniform_sphere(self, RAlim, DEClim, size=1):
        """Draw a uniform sample on a sphere

        Parameters
        ----------
        RAlim : tuple
            select Right Ascension between RAlim[0] and RAlim[1]
            units are degrees
        DEClim : tuple
            select Declination between DEClim[0] and DEClim[1]
        size : int (optional)
            the size of the random arrays to return (default = 1)

        Returns
        -------
        RA, DEC : ndarray
            the random sample on the sphere within the given limits.
            arrays have shape equal to size.
        """
        zlim = np.sin(np.pi * np.asarray(DEClim) / 180.)

        z = zlim[0] + (zlim[1] - zlim[0]) * np.random.random(size)
        DEC = (180. / np.pi) * np.arcsin(z)
        RA = RAlim[0] + (RAlim[1] - RAlim[0]) * np.random.random(size)

        return RA, DEC



class CheckFootprint(FootprintUtil):
    """
    contains checks for a certain coordinate and survey name
    """

    def select_survey(self, surveyname):
        """
        selects survey data to be read in
        """
        if surveyname == 'DES':
            filename = 'DESround13.txt'
        else:
            raise ValueError("survey %s you requested is not in the list" % (surveyname))
        filename = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..', '..', 'PackageData', filename))
        return filename

    def get_survey_data(self, filename):
        """
        returns raFoot and decFoot from the specified .txt file
        """
        # Read the file.
        f2 = open(filename, 'r')
        # read the whole file into a single variable, which is a list of every row of the file.
        lines = f2.readlines()
        f2.close()

        # initialize some variable to be lists:
        x1 = []
        y1 = []

        # scan the rows of the file stored in lines, and put the values into some variables:
        for line in lines:
            if '#' not in line:
                p = line.split()
                x1.append(float(p[0]))
                y1.append(float(p[1]))

        raFoot = np.array(x1)
        decFoot = np.array(y1)
        return raFoot,decFoot

    def check_footprint(self,ra,dec,surveyname = 'DES'):
        """
        main function which returns True, when coordinate is within the survey and False otherwise
        """
        filename = self.select_survey(surveyname)
        raFoot,decFoot = self.get_survey_data(filename)
        return self.insideFootprint(ra, dec, raFoot, decFoot)

