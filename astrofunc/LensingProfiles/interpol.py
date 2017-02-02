__author__ = 'sibirrer'

import scipy.interpolate
import numpy as np

import astrofunc.util as util

class Interpol_func(object):
    """
    class which uses an interpolation of a lens model and its first and second order derivatives
    """

    def function(self, x, y):
        if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_ = self.f_interp(y, x)
            return f_[0][0]
        else:
            x_axes, y_axes = util.get_axes(x, y)
            f_ = self.f_interp(y_axes, x_axes)
            f_ = util.image2array(f_)
        return f_

    def derivatives(self, x, y):
        """
        returns df/dx and df/dy of the function
        """
        if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_x = self.f_x_interp(y, x)
            f_y = self.f_y_interp(y, x)
            return f_x[0][0], f_y[0][0]
        else:
            x_, y_ = util.get_axes(x, y)
            f_x = self.f_x_interp(y_, x_)
            f_y = self.f_y_interp(y_, x_)
            f_x = util.image2array(f_x)
            f_y = util.image2array(f_y)
        return f_x, f_y

    def hessian(self, x, y):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_xx = self.f_xx_interp(y, x)
            f_yy = self.f_yy_interp(y, x)
            f_xy = self.f_xy_interp(y, x)
            return f_xx[0][0], f_yy[0][0], f_xy[0][0]
        else:
            x_, y_ = util.get_axes(x, y)
            f_xx = self.f_xx_interp(y_, x_)
            f_yy = self.f_yy_interp(y_, x_)
            f_xy = self.f_xy_interp(y_, x_)
            f_xx = util.image2array(f_xx)
            f_yy = util.image2array(f_yy)
            f_xy = util.image2array(f_xy)
        return f_xx, f_yy, f_xy

    def all(self, x, y):
        """
        returns f,f_x,f_y,f_xx, f_yy, f_xy
        """
        if type(x) == float or type(x) == int or type(x) == type(np.float64(1)) or len(x) <= 1:
            f_ = self.f_interp(y, x)
            f_x = self.f_x_interp(y, x)
            f_y = self.f_y_interp(y, x)
            f_xx = self.f_xx_interp(y, x)
            f_yy = self.f_yy_interp(y, x)
            f_xy = self.f_xy_interp(y, x)
            return f_[0][0], f_x[0][0], f_y[0][0], f_xx[0][0], f_yy[0][0], f_xy[0][0]
        else:
            x_, y_ = util.get_axes(x, y)
            f_ = self.f_interp(y_, x_)
            f_x = self.f_x_interp(y_, x_)
            f_y = self.f_y_interp(y_, x_)
            f_xx = self.f_xx_interp(y_, x_)
            f_yy = self.f_yy_interp(y_, x_)
            f_xy = self.f_xy_interp(y_, x_)
            f_ = util.image2array(f_)
            f_x = util.image2array(f_x)
            f_y = util.image2array(f_y)
            f_xx = util.image2array(f_xx)
            f_yy = util.image2array(f_yy)
            f_xy = util.image2array(f_xy)
        return f_, f_x, f_y, f_xx, f_yy, f_xy

    def do_interp(self, x_grid, y_grid, f_, f_x, f_y, f_xx, f_yy, f_xy):
        self.f_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_, kx=1, ky=1, s=0)
        self.f_x_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_x, kx=1, ky=1, s=0)
        self.f_y_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_y, kx=1, ky=1, s=0)
        self.f_xx_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xx, kx=1, ky=1, s=0)
        self.f_xy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_xy, kx=1, ky=1, s=0)
        self.f_yy_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, f_yy, kx=1, ky=1, s=0)


