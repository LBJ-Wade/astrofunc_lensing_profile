__author__ = 'sibirrer'

from scipy import fftpack
import numpy as np
import util


class Correlation(object):
    """
    class to analyse correlations in an image or in the residuals
    """
    def correlation_2D(self, image):
        """

        :param residuals:
        :return:
        """
        # Take the fourier transform of the image.
        F1 = fftpack.fft2(image)

        # Now shift the quadrants around so that low spatial frequencies are in
        # the center of the 2D fourier transformed image.
        F2 = fftpack.fftshift(F1)

        # Calculate a 2D power spectrum
        psd2D = np.abs(F2)

        # Calculate the azimuthally averaged 1D power spectrum
        psd1D = util.azimuthalAverage(psd2D)
        return psd1D, psd2D

    def random_1D(self, numPix):
        """
        same as corr1D put with a random distribution with mean 0 and sigma=1
        :return:
        """
        noise = util.array2image(np.random.normal(0, 1, numPix**2))**2
        psd1D_noise, psd2D_noise = self.correlation_2D(noise)
        return psd1D_noise, psd2D_noise