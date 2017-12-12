__author__ = 'sibirrer'

import astrofunc.util as Util
import scipy.ndimage.interpolation as interp
import numpy as np
import pytest
import numpy.testing as npt

def test_map_coord2pix():
    ra = 0
    dec = 0
    x_0 = 1
    y_0 = -1
    M = np.array([[1, 0], [0, 1]])
    x, y = Util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x == 1
    assert y == -1

    ra = [0, 1, 2]
    dec = [0, 2, 1]
    x, y = Util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x[0] == 1
    assert y[0] == -1
    assert x[1] == 2

    M = np.array([[0, 1], [1, 0]])
    x, y = Util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x[1] == 3
    assert y[1] == 0


def test_cart2polar():
    #singel 2d coordinate transformation
    center = np.array([0,0])
    x = 1
    y = 1
    r, phi = Util.cart2polar(x,y,center)
    assert r == np.sqrt(2) #radial part
    assert phi == np.arctan(1)
    #array of 2d coordinates
    center = np.array([0,0])
    x = np.array([1,2])
    y = np.array([1,1])

    r, phi = Util.cart2polar(x,y,center)
    assert r[0] == np.sqrt(2) #radial part
    assert phi[0] == np.arctan(1)

def test_polar2cart():
    #singel 2d coordinate transformation
    center = np.array([0,0])
    r = 1
    phi = np.pi
    x, y = Util.polar2cart(r, phi, center)
    assert x == -1
    assert abs(y) < 10e-14

def test_phi_q2_elliptisity():
    phi, q = 0, 1
    e1,e2 = Util.phi_q2_elliptisity(phi,q)
    assert e1 == 0
    assert e2 == 0

    phi, q = 1, 1
    e1,e2 = Util.phi_q2_elliptisity(phi,q)
    assert e1 == 0
    assert e2 == 0

    phi, q = 2.,0.95
    e1,e2 = Util.phi_q2_elliptisity(phi,q)
    assert e1 == -0.016760092842656733
    assert e2 == -0.019405192187382792

def test_elliptisity2phi_q():
    e1, e2 = 0.3,0
    phi,q = Util.elliptisity2phi_q(e1,e2)
    assert phi == 0
    assert q == 0.53846153846153844

def test_elliptisity2phi_q_symmetry():
    phi,q = 1.5, 0.8
    e1,e2 = Util.phi_q2_elliptisity(phi,q)
    phi_new,q_new = Util.elliptisity2phi_q(e1,e2)
    assert phi == phi_new
    assert q == q_new

    phi,q = -1.5, 0.8
    e1,e2 = Util.phi_q2_elliptisity(phi,q)
    phi_new,q_new = Util.elliptisity2phi_q(e1,e2)
    assert phi == phi_new
    assert q == q_new

    e1, e2 = 0.1, -0.1
    phi, q = Util.elliptisity2phi_q(e1, e2)
    e1_new, e2_new = Util.phi_q2_elliptisity(phi,q)
    npt.assert_almost_equal(e1, e1_new, decimal=10)
    npt.assert_almost_equal(e2, e2_new, decimal=10)

    e1, e2 = 2.8, -0.8
    phi, q = Util.elliptisity2phi_q(e1, e2)
    e1_new, e2_new = Util.phi_q2_elliptisity(phi,q)
    npt.assert_almost_equal(e1, e1_new, decimal=10)
    npt.assert_almost_equal(e2, e2_new, decimal=10)


def test_get_mask():
    x=np.linspace(0,10,100)
    y=np.linspace(0,10,100)
    center_x = 5
    center_y = 5
    r = 1
    mask = Util.get_mask(center_x,center_y,r,x,y)
    assert mask[0][0] == 1
    assert mask[5][5] == 0


def test_make_grid():
    numPix = 11
    deltapix = 1
    grid = Util.make_grid(numPix, deltapix)
    assert grid[0][0] == -5
    x_grid, y_grid = Util.make_grid(numPix, deltapix, subgrid_res=2.)
    assert x_grid[0] == -5.5


def test_grid_with_coords():
    numPix = 11
    deltaPix = 1.
    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = Util.make_grid_with_coordtransform(numPix, deltaPix, subgrid_res=1, left_lower=False)
    ra = 0
    dec = 0
    x, y = Util.map_coord2pix(ra, dec, x_at_radec_0, y_at_radec_0, Mcoord2pix)
    assert x == 5
    assert y == 5

    numPix = 11
    deltaPix = .1
    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = Util.make_grid_with_coordtransform(numPix, deltaPix, subgrid_res=1, left_lower=False)
    ra = 0
    dec = 0
    x, y = Util.map_coord2pix(ra, dec, x_at_radec_0, y_at_radec_0, Mcoord2pix)
    assert x == 5
    assert y == 5

    numPix = 11
    deltaPix = 1.
    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = Util.make_grid_with_coordtransform(numPix, deltaPix, subgrid_res=1, left_lower=False)
    x_, y_ = 0, 0
    ra, dec = Util.map_coord2pix(x_, y_, ra_at_xy_0, dec_at_xy_0, Mpix2coord)
    assert ra == -5
    assert dec == -5

    numPix = 11
    deltaPix = .1
    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = Util.make_grid_with_coordtransform(numPix, deltaPix, subgrid_res=1, left_lower=False)
    x_, y_ = 0, 0
    ra, dec = Util.map_coord2pix(x_, y_, ra_at_xy_0, dec_at_xy_0, Mpix2coord)
    assert ra == -.5
    assert dec == -.5
    x__, y__ = Util.map_coord2pix(ra, dec, x_at_radec_0, y_at_radec_0, Mcoord2pix)
    assert x__ == x_
    assert y__ == y_


def test_array2image():
    array = np.linspace(1, 100, 100)
    image = Util.array2image(array)
    assert image[9][9] == 100
    assert image[0][9] == 10


def test_image2array():
    image = np.zeros((10,10))
    image[1,2] = 1
    array = Util.image2array(image)
    assert array[12] == 1


def test_image2array2image():
    image = np.zeros((20, 10))
    nx, ny = np.shape(image)
    image[1, 2] = 1
    array = Util.image2array(image)
    image_new = Util.array2image(array, nx, ny)
    assert image_new[1, 2] == image[1, 2]


def test_get_axes():
    numPix = 11
    deltapix = 0.1
    x_grid, y_grid = Util.make_grid(numPix,deltapix)
    x_axes, y_axes = Util.get_axes(x_grid, y_grid)
    assert x_axes[0] == -0.5
    assert y_axes[0] == -0.5
    assert x_axes[1] == -0.4
    assert y_axes[1] == -0.4
    x_grid += 1
    x_axes, y_axes = Util.get_axes(x_grid, y_grid)
    assert x_axes[0] == 0.5
    assert y_axes[0] == -0.5


def test_symmetry():
    array = np.linspace(0,10,100)
    image = Util.array2image(array)
    array_new = Util.image2array(image)
    assert array_new[42] == array[42]

def test_cut_edges():
    image = np.zeros((51,51))
    image[25][25] = 1
    numPix = 21
    resized = Util.cut_edges(image, numPix)
    nx, ny = resized.shape
    assert nx == numPix
    assert ny == numPix
    assert resized[10][10] == 1


def test_displaceAbs():
    x = np.array([0,1,2])
    y = np.array([3,2,1])
    sourcePos_x = 1
    sourcePos_y = 2
    result = Util.displaceAbs(x, y, sourcePos_x, sourcePos_y)
    assert result[0] == np.sqrt(2)
    assert result[1] == 0


def test_add_layer2image_odd_odd():
    grid2d = np.zeros((101, 101))
    kernel = np.zeros((21, 21))
    kernel[10, 10] = 1
    x_pos = 50
    y_pos = 50
    added = Util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    #print added[45:56, 45:56]
    assert added[50, 50] == 1
    assert added[49, 49] == 0

    x_pos = 70
    y_pos = 95
    added = Util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)

    assert added[95, 70] == 1

    x_pos = 20
    y_pos = 45
    added = Util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[45, 20] == 1

    x_pos = 45
    y_pos = 20
    added = Util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[20, 45] == 1

    x_pos = 20
    y_pos = 55
    added = Util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    print added[50:61, 15:26]
    assert added[55, 20] == 1

    x_pos = 20
    y_pos = 100
    added = Util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[100, 20] == 1


def test_cutout_source():
    grid2d = np.zeros((20, 20))
    grid2d[7:9, 7:9] = 1
    kernel = Util.cutout_source(x_pos=7.5, y_pos=7.5, image=grid2d, kernelsize=5, shift=False)
    print kernel
    assert kernel[2, 2] == 1


def test_cutout_psf():
    """
    test whether a shifted psf can be reproduced sufficiently well
    :return:
    """
    kernel_size = 5
    image = np.zeros((10, 10))
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[2, 2] = 1
    shift_x = 0.5
    shift_y = 0
    x_c, y_c = 5, 5
    x_pos = x_c + shift_x
    y_pos = y_c + shift_y
    #kernel_shifted = interp.shift(kernel, [shift_y, shift_x], order=1)
    image = Util.add_layer2image(image, x_pos, y_pos, kernel, order=1)
    print(image)
    kernel_new = Util.cutout_source(x_pos=x_pos, y_pos=y_pos, image=image, kernelsize=kernel_size)
    print kernel_new, kernel
    npt.assert_almost_equal(kernel_new[2, 2], kernel[2, 2], decimal=2)


def test_de_shift():
    kernel_size = 5
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[2, 2] = 2
    shift_x = 0.48
    shift_y = 0.2
    kernel_shifted = interp.shift(kernel, [-shift_y, -shift_x], order=1)
    kernel_de_shifted = Util.de_shift_kernel(kernel_shifted, shift_x, shift_y, iterations=50)
    delta_max = np.max(kernel- kernel_de_shifted)
    assert delta_max < 0.01
    npt.assert_almost_equal(kernel_de_shifted[2, 2], kernel[2, 2], decimal=2)

    kernel_size = 5
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[2, 2] = 2
    shift_x = 1.48
    shift_y = 0.2
    kernel_shifted = interp.shift(kernel, [-shift_y, -shift_x], order=1)
    kernel_de_shifted = Util.de_shift_kernel(kernel_shifted, shift_x, shift_y, iterations=50)
    delta_max = np.max(kernel- kernel_de_shifted)
    print kernel_de_shifted - kernel
    assert delta_max < 0.01
    npt.assert_almost_equal(kernel_de_shifted[2, 2], kernel[2, 2], decimal=2)


def test_shift_long_dist():
    """
    input is a shifted kernel by more than 1 pixel
    :return:
    """

    kernel_size = 9
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[4, 4] = 2.
    shift_x = 2.
    shift_y = 1.
    input_kernel = interp.shift(kernel, [-shift_y, -shift_x], order=1)
    old_style_kernel = interp.shift(input_kernel, [shift_y, shift_x], order=1)
    shifted_new = Util.de_shift_kernel(input_kernel, shift_x, shift_y)
    print shifted_new - old_style_kernel
    assert kernel[3, 2] == shifted_new[3, 2]
    assert np.max(old_style_kernel - shifted_new) < 0.01


def test_pixel_kernel():
    # point source kernel
    kernel_size = 9
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[4, 4] = 1.
    pixel_kernel = Util.pixel_kernel(point_source_kernel=kernel, subgrid_res=1)
    assert pixel_kernel[4, 4] == kernel[4, 4]

    pixel_kernel = Util.pixel_kernel(point_source_kernel=kernel, subgrid_res=11)
    print pixel_kernel
    npt.assert_almost_equal(pixel_kernel[4, 4], 0.3976, decimal=3)


def test_mk_array():
    variable = 1.
    output = Util.mk_array(variable)
    assert output[0] == 1
    variable = [1,2,3]
    output = Util.mk_array(variable)
    assert output[0] == 1


def test_get_distance():
    x_mins = Util.mk_array(1.)
    y_mins = Util.mk_array(1.)
    x_true = Util.mk_array(0.)
    y_true = Util.mk_array(0.)
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 2

    x_mins = Util.mk_array([1.,2])
    y_mins = Util.mk_array([1.,1])
    x_true = Util.mk_array(0.)
    y_true = Util.mk_array(0.)
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 10000000000

    x_mins = Util.mk_array([1.,2])
    y_mins = Util.mk_array([1.,1])
    x_true = Util.mk_array([0.,1])
    y_true = Util.mk_array([0.,2])
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 6

    x_mins = Util.mk_array([1.,2,0])
    y_mins = Util.mk_array([1.,1,0])
    x_true = Util.mk_array([0.,1,1])
    y_true = Util.mk_array([0.,2,1])
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 2


def test_phi_gamma_ellipticity():
    phi = -1.
    gamma = 0.1
    e1, e2 = Util.phi_gamma_ellipticity(phi, gamma)
    print(e1, e2, 'e1, e2')
    phi_out, gamma_out = Util.ellipticity2phi_gamma(e1, e2)
    assert phi == phi_out
    assert gamma == gamma_out


def test_selectBest():
    array = np.array([4,3,6,1,3])
    select = np.array([2,4,7,3,3])
    numSelect = 4
    array_select = Util.selectBest(array, select, numSelect, highest=True)
    assert array_select[0] == 6
    assert array_select[3] == 1


def test_add_background():
    image = np.ones((10, 10))
    sigma_bkgd = 1.
    image_noisy = Util.add_background(image, sigma_bkgd)
    assert abs(np.sum(image_noisy)) < np.sqrt(np.sum(image)*sigma_bkgd)*3


def test_add_poisson():
    image = np.ones((10, 10))
    exp_time = 100.
    poisson = Util.add_poisson(image, exp_time)
    assert abs(np.sum(poisson)) < np.sqrt(np.sum(image)/exp_time)*3


def test_rotateImage():
    img = np.zeros((5, 5))
    img[2, 2] = 1
    img[1, 2] = 0.5

    angle = 360
    im_rot = Util.rotateImage(img, angle)
    npt.assert_almost_equal(im_rot[1, 2], 0.5, decimal=10)
    npt.assert_almost_equal(im_rot[2, 2], 1., decimal=10)
    npt.assert_almost_equal(im_rot[2, 1], 0., decimal=10)

    angle = 360./2
    im_rot = Util.rotateImage(img, angle)
    print img
    print im_rot
    npt.assert_almost_equal(im_rot[1, 2], 0., decimal=10)
    npt.assert_almost_equal(im_rot[2, 2], 1., decimal=10)
    npt.assert_almost_equal(im_rot[3, 2], 0.5, decimal=10)

    angle = 360./4
    im_rot = Util.rotateImage(img, angle)
    print img
    print im_rot
    npt.assert_almost_equal(im_rot[1, 2], 0., decimal=10)
    npt.assert_almost_equal(im_rot[2, 2], 1., decimal=10)
    npt.assert_almost_equal(im_rot[2, 1], 0.5, decimal=10)

    angle = 360./8
    im_rot = Util.rotateImage(img, angle)
    print img
    print im_rot
    npt.assert_almost_equal(im_rot[1, 2], 0.23931518624017051, decimal=10)
    npt.assert_almost_equal(im_rot[2, 2], 1., decimal=10)
    npt.assert_almost_equal(im_rot[2, 1], 0.23931518624017073, decimal=10)


def test_neighborSelect():
    a = np.ones(100)
    a[41] = 0
    x = np.linspace(0,99,100)
    y = np.linspace(0,99,100)
    x_mins, y_mins, values = Util.neighborSelect(a, x, y)
    assert x_mins[0] == 41
    assert y_mins[0] == 41
    assert values[0] == 0


def test_averaging2():
    grid = np.ones((100, 100))
    grid_smoothed = Util.averaging2(grid, numGrid=100, numPix=50)
    assert grid_smoothed[0][0] == 1


def test_re_size_array():
    numPix = 9
    kernel = np.zeros((numPix, numPix))
    kernel[(numPix-1)/2, (numPix-1)/2] = 1
    subgrid_res = 2
    input_values = kernel
    x_in = np.linspace(0, 1, numPix)
    x_out = np.linspace(0, 1, numPix*subgrid_res)
    out_values = Util.re_size_array(x_in, x_in, input_values, x_out, x_out)
    kernel_out = out_values
    assert kernel_out[(numPix*subgrid_res-1)/2, (numPix*subgrid_res-1)/2] == 0.58477508650519028


def test_make_subgrid():
    numPix = 101
    deltapix = 1
    x_grid, y_grid = Util.make_grid(numPix, deltapix, subgrid_res=1)
    x_sub_grid, y_sub_grid = Util.make_subgrid(x_grid, y_grid, subgrid_res=2)
    assert np.sum(x_grid) == 0
    assert x_sub_grid[0] == -50.25
    assert y_sub_grid[17] == -50.25

    x_sub_grid_new, y_sub_grid_new = Util.make_subgrid(x_grid, y_grid, subgrid_res=4)
    assert x_sub_grid_new[0] == -50.375


def test_re_size2():
    kwargs = {'numPix': 50}
    grid = np.ones((100, 100))
    grid_smoothed = Util.re_size_grid(grid, **kwargs)
    assert grid_smoothed[0][0] == 1


def test_re_size():
    grid = np.zeros((200, 100))
    grid[100, 50] = 4
    grid_small = Util.re_size(grid, factor=2)
    assert grid_small[50][25] == 1


def test_symmetry_average():
    image = np.zeros((5,5))
    image[2, 3] = 1
    symmetry = 2
    img_sym = Util.symmetry_average(image, symmetry)
    npt.assert_almost_equal(img_sym[2, 1], 0.5, decimal=10)


def test_fwhm2sigma():
    fwhm = 0.5
    sigma = Util.fwhm2sigma(fwhm)
    assert sigma == fwhm/ (2 * np.sqrt(2 * np.log(2)))


def test_fwhm_kerne():
    x_grid, y_gird = Util.make_grid(101, 1)
    sigma = 20
    from astrofunc.LightProfiles.gaussian import Gaussian
    gaussian = Gaussian()
    flux = gaussian.function(x_grid, y_gird, amp=1, sigma_x=sigma, sigma_y=sigma)
    kernel = Util.array2image(flux)
    kernel = Util.kernel_norm(kernel)
    fwhm_kernel = Util.fwhm_kernel(kernel)
    fwhm = Util.sigma2fwhm(sigma)
    npt.assert_almost_equal(fwhm/fwhm_kernel, 1, 2)


if __name__ == '__main__':
    pytest.main()