__author__ = 'Simon Birrer'

"""
this file contains standard routines
"""

from collections import namedtuple
import numpy as np
import scipy.ndimage.interpolation as interp
import scipy.signal.signaltools as signaltools
import scipy
from numpy import linspace, meshgrid
import copy
import mpmath



def dictionary_to_namedtuple(dictionary):
    dictionary.pop("__name__", None)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def map_coord2pix(ra, dec, x_0, y_0, M):
    """
    this routines performs a linear transformation between two coordinate systems. Mainly used to transform angular
    into pixel coordinates in an image
    :param ra: ra coordinates
    :param dec: dec coordinates
    :param x_0: pixel value in x-axis of ra,dec = 0,0
    :param y_0: pixel value in y-axis of ra,dec = 0,0
    :param M: 2x2 matrix to transform angular to pixel coordinates
    :return: transformed coordnate systems of input ra and dec
    """
    x, y = M.dot(np.array([ra, dec]))
    return x + x_0, y + y_0


def cart2polar(x, y, center=np.array([0, 0])):
    """
    transforms cartesian coords [x,y] into polar coords [r,phi] in the frame of the lense center

    :param coord: set of coordinates
    :type coord: array of size (n,2)
    :param center: rotation point
    :type center: array of size (2)
    :returns:  array of same size with coords [r,phi]
    :raises: AttributeError, KeyError
    """
    coordShift_x = x - center[0]
    coordShift_y = y - center[1]
    r = np.sqrt(coordShift_x**2+coordShift_y**2)
    phi = np.arctan2(coordShift_y, coordShift_x)
    return r, phi


def polar2cart(r, phi, center):
    """
    transforms polar coords [r,phi] into cartesian coords [x,y] in the frame of the lense center

    :param coord: set of coordinates
    :type coord: array of size (n,2)
    :param center: rotation point
    :type center: array of size (2)
    :returns:  array of same size with coords [x,y]
    :raises: AttributeError, KeyError
    """
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x - center[0], y - center[1]


def array2image(array, nx=0, ny=0):
    """
    returns the information contained in a 1d array into an n*n 2d array (only works when lenght of array is n**2)

    :param array: image values
    :type array: array of size n**2
    :returns:  2d array
    :raises: AttributeError, KeyError
    """
    if nx == 0 or ny == 0:
        n = int(np.sqrt(len(array)))
        if n**2 != len(array):
            raise ValueError("lenght of input array given as %s is not square of integer number!" %(len(array)))
        nx, ny = n, n
    image = array.reshape(int(nx), int(ny))
    return image


def image2array(image):
    """
    returns the information contained in a 2d array into an n*n 1d array

    :param array: image values
    :type array: array of size (n,n)
    :returns:  1d array
    :raises: AttributeError, KeyError
    """
    nx, ny = image.shape  # find the size of the array
    imgh = np.reshape(image, nx*ny)  # change the shape to be 1d
    return imgh


def make_grid(numPix, deltapix, subgrid_res=1, left_lower=False):
    """

    :param numPix: number of pixels per axis
    :param deltapix: pixel size
    :param subgrid_res: sub-pixel resolution (default=1)
    :return: x, y position information in two 1d arrays
    """

    numPix_eff = numPix*subgrid_res
    deltapix_eff = deltapix/float(subgrid_res)
    a = np.arange(numPix_eff)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    if left_lower is True:
        x_grid = matrix[:, 0]*deltapix
        y_grid = matrix[:, 1]*deltapix
    else:
        x_grid = (matrix[:, 0] - (numPix_eff-1)/2.)*deltapix_eff
        y_grid = (matrix[:, 1] - (numPix_eff-1)/2.)*deltapix_eff
    shift = (subgrid_res-1)/(2.*subgrid_res)*deltapix
    return x_grid - shift, y_grid - shift


def make_grid_transformed(numPix, Mpix2Angle):
    """
    returns grid with linear transformation (deltaPix and rotation)
    :param numPix: number of Pixels
    :param Mpix2Angle: 2-by-2 matrix to mat a pixel to a coordinate
    :return: coordinate grid
    """
    x_grid, y_grid = make_grid(numPix, deltapix=1)
    ra_grid, dec_grid = map_coord2pix(x_grid, y_grid, 0, 0, Mpix2Angle)
    return ra_grid, dec_grid


def make_grid_with_coordtransform(numPix, deltapix, subgrid_res=1, left_lower=False):
    """
    same as make_grid routine, but returns the transformaton matrix and shift between coordinates and pixel
    :param numPix:
    :param deltapix:
    :param subgrid_res:
    :param left_lower:
    :return:
    """
    numPix_eff = numPix*subgrid_res
    deltapix_eff = deltapix/float(subgrid_res)
    a = np.arange(numPix_eff)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    if left_lower is True:
        x_grid = matrix[:, 0]*deltapix
        y_grid = matrix[:, 1]*deltapix
    else:
        x_grid = (matrix[:, 0] - (numPix_eff-1)/2.)*deltapix_eff
        y_grid = (matrix[:, 1] - (numPix_eff-1)/2.)*deltapix_eff
    shift = (subgrid_res-1)/(2.*subgrid_res)*deltapix
    x_grid -= shift
    y_grid -= shift
    ra_at_xy_0 = x_grid[0]
    dec_at_xy_0 = y_grid[0]
    x_at_radec_0 = (numPix_eff-1)/2.
    y_at_radec_0 = (numPix_eff - 1) / 2.
    Mpix2coord = np.array([[deltapix_eff, 0], [0, deltapix_eff]])
    Mcoord2pix = np.linalg.inv(Mpix2coord)
    return x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix


def get_axes(x, y):
    """
    computes the axis x and y of a given 2d grid
    :param x:
    :param y:
    :return:
    """
    n=int(np.sqrt(len(x)))
    if n**2 != len(x):
        raise ValueError("lenght of input array given as %s is not square of integer number!" % (len(x)))
    x_image = x.reshape(n,n)
    y_image = y.reshape(n,n)
    x_axes = x_image[0,:]
    y_axes = y_image[:,0]
    return x_axes, y_axes


def averaging(grid, numGrid, numPix):
    """
    resize 2d pixel grid with numGrid to numPix and averages over the pixels
    :param grid: higher resolution pixel grid
    :param numGrid:
    :param numPix:
    :return:
    """

    Nbig = numGrid
    Nsmall = numPix
    small = grid.reshape([Nsmall, Nbig/Nsmall, Nsmall, Nbig/Nsmall]).mean(3).mean(1)
    return small


def averaging2(grid, numGrid, numPix):
    """

    :param grid:
    :param numGrid:
    :param numPix:
    :return:
    """
    from PIL import Image
    width_org, height_org = grid.shape
    factor = numPix/float(numGrid)
    width = int(width_org * factor)
    height = int(height_org * factor)
    im = Image.fromarray(grid)
    img_anti = im.resize((width, height), Image.ANTIALIAS)
    return np.array(img_anti)


def phi_gamma_ellipticity(phi, gamma):
    """

    :param phi: angel
    :param gamma: ellipticity
    :return:
    """
    e1 = gamma*np.cos(2*phi)
    e2 = gamma*np.sin(2*phi)
    return e1, e2


def ellipticity2phi_gamma(e1, e2):
    """
    :param e1: ellipticity component
    :param e2: ellipticity component
    :return: angle and abs value of ellipticity
    """
    phi = np.arctan2(e2, e1)/2
    gamma = np.sqrt(e1**2+e2**2)
    return phi, gamma


def phi_q2_elliptisity(phi, q):
    """

    :param phi:
    :param q:
    :return:
    """
    e1 = (1.-q)/(1.+q)*np.cos(2*phi)
    e2 = (1.-q)/(1.+q)*np.sin(2*phi)
    return e1, e2


def elliptisity2phi_q(e1,e2):
    """
    :param e1:
    :param e2:
    :return:
    """
    phi = np.arctan2(e2, e1)/2
    c = np.sqrt(e1**2+e2**2)
    q = (1-c)/(1+c)
    return phi, q


def get_mask(center_x, center_y, r, x, y):
    """

    :param center: 2D coordinate of center position of circular mask
    :param r: radius of mask in pixel values
    :param data: data image
    :return:
    """
    x_shift = x - center_x
    y_shift = y - center_y
    R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
    mask = np.empty_like(R)
    mask[R > r] = 1
    mask[R <= r] = 0
    n = int(np.sqrt(len(x)))
    mask_2d = mask.reshape(n, n)
    return mask_2d


def mask_sphere(x, y, center_x, center_y, r):
    """

    :param center: 2D coordinate of center position of circular mask
    :param r: radius of mask in pixel values
    :param data: data image
    :return:
    """
    x_shift = x - center_x
    y_shift = y - center_y
    R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
    mask = np.empty_like(R)
    mask[R > r] = 0
    mask[R <= r] = 1
    return mask


def mask_half_moon(x, y, center_x, center_y, r_in, r_out, phi0=0, delta_phi=2*np.pi):
    """

    :param x:
    :param y:
    :param center_x:
    :param center_y:
    :param r_in:
    :param r_out:
    :param phi:
    :param delta_phi:
    :return:
    """
    x_shift = x - center_x
    y_shift = y - center_y
    R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
    phi = np.arctan2(x_shift, y_shift)
    #phi = np.abs(phi)
    phi_min = phi0 - delta_phi/2.
    phi_max = phi0 + delta_phi/2.
    mask = np.zeros_like(x)
    if phi_max > phi_min:
        mask[(R < r_out) & (R > r_in) & (phi > phi_min) & (phi < phi_max)] = 1
    else:
        mask[(R < r_out) & (R > r_in) & (phi > phi_max)] = 1
        mask[(R < r_out) & (R > r_in) & (phi < phi_min)] = 1
    return mask


def rotateImage(img, angle):
    """


    :param img:
    :param angle:
    :return:
    """
    imgR = scipy.ndimage.rotate(img, angle, reshape=False)
    return imgR


def compare(model, data, sigma, poisson):
    """

    :param model: model 2d image
    :param data: data 2d image
    :param sigma: minimal noise level of background (float>0 or as image)
    :return: X^2 value if images have same size
    """
    deltaIm = (data-model)**2
    relDeltaIm = deltaIm/(sigma**2 + np.abs(model)/poisson)
    X2_estimate = np.sum(relDeltaIm)
    return X2_estimate


def cut_edges(image, numPix):
    """
    cuts out the edges of a 2d image and returns re-sized image to numPix
    :param image: 2d numpy array
    :param numPix:
    :return:
    """
    nx, ny = image.shape
    if nx < numPix or ny < numPix:
        print('WARNING: image can not be resized.')
        return image
    if nx % 2 == 0 or ny % 2 == 0 or numPix % 2 == 0:
        #pass
        print("WARNING: image or cutout side are even number. This routine only works for odd numbers %s %s %s"
                         % (nx, ny, numPix))
    cx = int((nx-1)/2)
    cy = int((ny-1)/2)
    d = int((numPix-1)/2)
    if nx % 2 == 0:
        cx += 1
    if ny % 2 == 0:
        cy += 1
    resized = image[cx-d:cx+d+1, cy-d:cy+d+1]
    return copy.deepcopy(resized)


def displaceAbs(x, y, sourcePos_x, sourcePos_y):
    """
    calculates a grid of distances to the observer in angel

    :param mapped_cartcoord: mapped cartesian coordinates
    :type mapped_cartcoord: numpy array (n,2)
    :param sourcePos: source position
    :type sourcePos: numpy vector [x0,y0]
    :returns:  array of displacement
    :raises: AttributeError, KeyError
    """
    x_mapped = x - sourcePos_x
    y_mapped = y - sourcePos_y
    absmapped = np.sqrt(x_mapped**2+y_mapped**2)
    return absmapped


def add_layer2image(grid2d, x_pos, y_pos, kernel, order=1):
    """
    makes a point source on a grid with shifted PSF
    :param x_pos:
    :param y_pos:
    :return:
    """
    num_x, num_y = np.shape(grid2d)
    x_int = int(round(x_pos))
    y_int = int(round(y_pos))
    shift_x = x_int - x_pos
    shift_y = y_int - y_pos
    kernel_shifted = interp.shift(kernel, [-shift_y, -shift_x], order=order)
    k_x, k_y = np.shape(kernel)
    k_l2_x = int((k_x - 1) / 2)
    k_l2_y = int((k_y - 1) / 2)

    min_x = np.maximum(0, x_int-k_l2_x)
    min_y = np.maximum(0, y_int-k_l2_y)
    max_x = np.minimum(num_x, x_int+k_l2_x + 1)
    max_y = np.minimum(num_y, y_int+k_l2_y + 1)

    min_xk = np.maximum(0, -x_int + k_l2_x)
    min_yk = np.maximum(0, -y_int + k_l2_y)
    max_xk = np.minimum(k_x, -x_int + k_l2_x + num_x)
    max_yk = np.minimum(k_y, -y_int + k_l2_y + num_y)
    if min_x >= max_x or min_y >= max_y or min_xk >= max_xk or min_yk >= max_yk or (max_x-min_x != max_xk-min_xk) or (max_y-min_y != max_yk-min_yk):
        return grid2d
    kernel_re_sized = kernel_shifted[min_yk:max_yk, min_xk:max_xk]
    new = grid2d.copy()

    new[min_y:max_y, min_x:max_x] += kernel_re_sized
    return new


def cutout_source(x_pos, y_pos, image, kernelsize, shift=True):
    """
    cuts out point source (e.g. PSF estimate) out of image and shift it to the center of a pixel
    :param x_pos:
    :param y_pos:
    :param image:
    :param kernelsize:
    :return:
    """
    if kernelsize%2 == 0:
        raise ValueError("even pixel number kernel size not supported!")
    x_int = int(round(x_pos))
    y_int = int(round(y_pos))
    n = len(image)
    d = (kernelsize - 1)/2
    x_max = np.minimum(x_int + d + 1, n)
    x_min = np.maximum(x_int - d, 0)
    y_max = np.minimum(y_int + d + 1, n)
    y_min = np.maximum(y_int - d, 0)
    image_cut = copy.deepcopy(image[y_min:y_max, x_min:x_max])
    shift_x = x_int - x_pos
    shift_y = y_int - y_pos
    if shift is True:
        kernel_final = de_shift_kernel(image_cut, shift_x, shift_y)
        #kernel_shifted = interp.shift(image_cut, [shift_y, shift_x], order=order)
        #print(kernel_shifted, 'kernel_shifted')
        #kernel_inv_shifted = interp.shift(copy.deepcopy(kernel_shifted), [-shift_y, -shift_x], order=order)
        #print(kernel_inv_shifted, 'kernel_inv_shifted')
        #kernel_final = kernel_shifted + image_cut - kernel_inv_shifted
    else:
        kernel_final = image_cut
    return kernel_final


def de_shift_kernel(kernel, shift_x, shift_y, iterations=20):
    """

    :param kernel:
    :param shift_x:
    :param shift_y:
    :return:
    """
    n = len(kernel)
    kernel_new = np.zeros((n+2, n+2)) + (kernel[0, 0] + kernel[0, -1] + kernel[-1, 0] + kernel[-1, -1]) / 4.
    kernel_new[1:-1, 1:-1] = kernel
    int_shift_x = int(round(shift_x))
    frac_x_shift = shift_x - int_shift_x
    int_shift_y = int(round(shift_y))
    frac_y_shift = shift_y - int_shift_y
    kernel_init = copy.deepcopy(kernel_new)
    kernel_init_shifted = copy.deepcopy(interp.shift(kernel_init, [int_shift_y, int_shift_x], order=1))
    kernel_new = interp.shift(kernel_new, [int_shift_y, int_shift_x], order=1)
    norm = np.sum(kernel_new)
    for i in range(iterations):
        kernel_shifted_inv = interp.shift(kernel_new, [-frac_y_shift, -frac_x_shift], order=1)
        delta = kernel_init_shifted - kernel_norm(kernel_shifted_inv) * norm
        kernel_new += delta
        kernel_new = kernel_norm(kernel_new) * norm
    return kernel_new[1:-1, 1:-1]


def kernel_norm(kernel):
    """

    :param kernel:
    :return: normalisation of the psf kernel
    """
    norm = np.sum(np.array(kernel))
    kernel /= norm
    return kernel


def subgrid_kernel(kernel, subgrid_res):
    """
    creates a higher resolution kernel with subgrid resolution as an interpolation of the original kernel
    :param kernel: initial kernel
    :param subgrid_res: subgrid resolution required
    :return: kernel with higher resolution (larger)
        """
    numPix = len(kernel)
    x_in = np.linspace(0, 1, numPix)
    x_out = np.linspace(0, 1, numPix * subgrid_res)
    out_values = re_size_array(x_in, x_in, kernel, x_out, x_out)
    kernel_subgrid = out_values
    kernel_subgrid = kernel_norm(kernel_subgrid)
    return kernel_subgrid


def kernel_pixelsize_change(kernel, deltaPix_in, deltaPix_out):
    """
    change the pixel size of a given kernel
    :param kernel:
    :param deltaPix_in:
    :param deltaPix_out:
    :return:
    """
    numPix = len(kernel)
    numPix_new = int(round(numPix * deltaPix_in/deltaPix_out))
    if numPix_new % 2 == 0:
        numPix_new -= 1
    x_in = np.linspace(-(numPix-1)/2*deltaPix_in, (numPix-1)/2*deltaPix_in, numPix)
    x_out = np.linspace(-(numPix_new-1)/2*deltaPix_out, (numPix_new-1)/2*deltaPix_out, numPix_new)
    kernel_out = re_size_array(x_in, x_in, kernel, x_out, x_out)
    kernel_out = kernel_norm(kernel_out)
    return kernel_out


def pixel_kernel(point_source_kernel, subgrid_res=7):
    """
    converts a pixelised kernel of a point source to a kernel representing a uniform extended pixel
    :param point_source_kernel:
    :param subgrid_res:
    :return: convolution kernel for an extended pixel
    """
    kernel_subgrid = subgrid_kernel(point_source_kernel, subgrid_res)
    kernel_size = len(point_source_kernel)
    kernel_pixel = np.zeros((kernel_size*subgrid_res, kernel_size*subgrid_res))
    for i in range(subgrid_res):
        k_x = int((kernel_size-1) / 2 * subgrid_res + i)
        for j in range(subgrid_res):
            k_y = int((kernel_size-1) / 2 * subgrid_res + j)
            kernel_pixel = add_layer2image(kernel_pixel, k_x, k_y, kernel_subgrid)
    kernel_pixel = averaging(kernel_pixel, numGrid=kernel_size*subgrid_res, numPix=kernel_size)
    return kernel_norm(kernel_pixel)


def get_distance(x_mins, y_mins, x_true, y_true):
    """

    :param x_mins:
    :param y_mins:
    :param x_true:
    :param y_true:
    :return:
    """
    if len(x_mins) != len(x_true):
        return 10**10
    dist = 0
    x_true_list = np.array(x_true)
    y_true_list = np.array(y_true)

    for i in range(0,len(x_mins)):
        dist_list = (x_mins[i] - x_true_list)**2 + (y_mins[i] - y_true_list)**2
        dist += min(dist_list)
        k = np.where(dist_list == min(dist_list))
        if type(k) != int:
            k = k[0]
        x_true_list = np.delete(x_true_list, k)
        y_true_list = np.delete(y_true_list, k)
    return dist


def compare_distance(x_mapped, y_mapped):
    """

    :param x_mapped: array of x-positions of remapped catalogue image
    :param y_mapped: array of y-positions of remapped catalogue image
    :return: sum of distance square of positions
    """
    X2 = 0
    for i in range(0, len(x_mapped)-1):
        for j in range(i+1, len(x_mapped)):
            dx = x_mapped[i]-x_mapped[j]
            dy = y_mapped[i]-y_mapped[j]
            X2 += dx**2+dy**2
    return X2


def min_square_dist(x_1, y_1, x_2, y_2):
    """
    return minimum of quadratic distance of pairs (x1, y1) to pairs (x2, y2)
    :param x_1:
    :param y_1:
    :param x_2:
    :param y_2:
    :return:
    """
    dist = np.zeros_like(x_1)
    for i in range(len(x_1)):
        dist[i] = np.min((x_1[i] - x_2)**2 + (y_1[i] - y_2)**2)
    return dist


def mk_array(input_var):
    """This functions makes sure that the input is a numpy array. If it is
    a recognised format (float, array or list) the output will be a numpy array"""

    if type(input_var) is float:
        output_var = np.array([input_var]) # turning a into a numpy array
    elif type(input_var) is type(np.float64(1)):
        output_var = np.array([np.asscalar(input_var)]) # turning a into a numpy array
    elif type(input_var) == type(np.array([])):
        output_var = input_var
    elif type(input_var) == list:
        output_var = np.array(input_var)
    else:
        print('input type for a not recognised. please use either float or numpy array')
        print(type(input_var))
        return 'ERROR!'

    return output_var


def mk_array_2p(input1, input2):
    """This functions makes sure that the input is a numpy array. If it is
    a recognised format (float, array or list) the output will be a numpy array"""

    if type(input1) is float:
        output1 = np.array([input1]) # turning a into a numpy array
    elif type(input1) is type(np.float64(1)):
        output1 = np.array([input2]) # turning a into a numpy array
    elif type(input1) == type(np.array([])):
        output1 = input1
    elif type(input1) == list:
        output1 = np.array(input1)
    else:
        print('input type for a not recognised. please use either float or numpy array')
        return 'ERROR!'

    if type(input2) is float:
        output2 = np.array([input2]) # turning a into a numpy array
    elif type(input2) is type(np.float64(1)):
        output2 = np.array([input2]) # turning a into a numpy array
    elif type(input2) == type(np.array([])):
        output2 = input2
    elif type(input2) == list:
        output2 = np.array(input2)
    else:
        print('input type for a not recognised. please use either float or numpy array')
        return 'ERROR!'

    output12_format = np.zeros([len(output1),len(output2)])

    return output1,output2,output12_format


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def selectBest(array, criteria, numSelect, highest=True):
    """

    :param array: numpy array to be selected from
    :param criteria: criteria of selection
    :param highest: bool, if false the lowest will be selected
    :param numSelect: number of elements to be selected
    :return:
    """
    n = len(array)
    m = len(criteria)
    if n != m:
        raise ValueError('Elements in array (%s) not equal to elements in criteria (%s)' % (n, m))
    if n < numSelect:
        return array
    array_sorted = array[criteria.argsort()]
    if highest:
        result = array_sorted[n-numSelect:]
    else:
        result = array_sorted[0:numSelect]
    return result[::-1]


def compute_lower_upper_errors(sample, num_sigma=1):
    """
    computes the upper and lower sigma from the median value.
    This functions gives good error estimates for skewed pdf's
    :param sample: 1-D sample
    :return: median, lower_sigma, upper_sigma
    """
    if num_sigma > 3:
        raise ValueError("Number of sigma-constraints restircted to three. %s not valid" % num_sigma)
    num = len(sample)
    num_threshold1 = int(round((num-1)*0.833))
    num_threshold2 = int(round((num-1)*0.977249868))
    num_threshold3 = int(round((num-1)*0.998650102))

    mean = np.mean(sample)
    sorted_sample = np.sort(sample)
    if num_sigma > 0:
        upper_sigma1 = sorted_sample[num_threshold1-1]
        lower_sigma1 = sorted_sample[num-num_threshold1-1]
    else:
        return mean, [[]]
    if num_sigma > 1:
        upper_sigma2 = sorted_sample[num_threshold2-1]
        lower_sigma2 = sorted_sample[num-num_threshold2-1]
    else:
        return mean, [[mean-lower_sigma1, upper_sigma1-mean]]
    if num_sigma > 2:
        upper_sigma3 = sorted_sample[num_threshold3-1]
        lower_sigma3 = sorted_sample[num-num_threshold3-1]
        return mean, [[mean-lower_sigma1, upper_sigma1-mean], [mean-lower_sigma2, upper_sigma2-mean],
                      [mean-lower_sigma3, upper_sigma3-mean]]
    else:
        return mean, [[mean-lower_sigma1, upper_sigma1-mean], [mean-lower_sigma2, upper_sigma2-mean]]


def add_background(image, sigma_bkd):
    """
    adds background noise to image
    :param image: pixel values of image
    :param sigma_bkd: background noise (sigma)
    :return:
    """
    if sigma_bkd < 0:
        raise ValueError("Sigma background is smaller than zero! Please use positive values.")
    nx, ny = np.shape(image)
    background = np.random.randn(nx, ny) * sigma_bkd
    return background


def add_poisson(image, exp_time):
    """
    adds a poison (or Gaussian) distributed noise with mean given by surface brightness
    :param image: pixel values (photon counts per unit exposure time)
    :param exp_time: exposure time
    :return: Poisson noise realization of input image
    """
    """
    adds a poison (or Gaussian) distributed noise with mean given by surface brightness
    """
    if isinstance(exp_time, int) or isinstance(exp_time, float):
        if exp_time <= 0:
            exp_time = 1
    else:
        mean_exp_time = np.mean(exp_time)
        exp_time[exp_time < mean_exp_time/10] = mean_exp_time/10
    sigma = np.sqrt(np.abs(image)/exp_time) # Gaussian approximation for Poisson distribution, normalized to exposure time
    nx, ny = np.shape(image)
    poisson = np.random.randn(nx, ny) * sigma
    return poisson


def grid(x, y, z, resX=100, resY=100):
    from matplotlib.mlab import griddata
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp="linear")
    X, Y = meshgrid(xi, yi)
    return X, Y, Z


def circle(x, y, center_x, center_y, radius):
    """
    uniform density circle
    :param x: x-coordinates
    :param y: y-coordinates
    :param center_x: center of x-coordinates
    :param center_y: center of y-coordinates
    :param radius: radius of circle
    :return:
    """
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    circle_draw = np.zeros_like(r)
    circle_draw[r < radius] = 1
    return circle_draw


def points_on_circle(radius, points):
    """
    returns a set of uniform points around a circle
    :param radius: radius of the circle
    :param points: number of points on the circle
    :return:
    """
    angle = np.linspace(0, 2*np.pi, points)
    x_coord = np.cos(angle)*radius
    y_coord = np.sin(angle)*radius
    return x_coord, y_coord


def neighborSelect(a, x, y):
    """
    finds (local) minima in a 2d grid

    :param a: 1d array of displacements from the source positions
    :type a: numpy array with length numPix**2 in float
    :returns:  array of indices of local minima, values of those minima
    :raises: AttributeError, KeyError
    """
    dim = int(np.sqrt(len(a)))
    values = []
    x_mins = []
    y_mins = []
    for i in range(dim+1,len(a)-dim-1):
        if (a[i] < a[i-1]
            and a[i] < a[i+1]
            and a[i] < a[i-dim]
            and a[i] < a[i+dim]
            and a[i] < a[i-(dim-1)]
            and a[i] < a[i-(dim+1)]
            and a[i] < a[i+(dim-1)]
            and a[i] < a[i+(dim+1)]):
                if(a[i] < a[(i-2*dim-1)%dim**2]
                    and a[i] < a[(i-2*dim+1)%dim**2]
                    and a[i] < a[(i-dim-2)%dim**2]
                    and a[i] < a[(i-dim+2)%dim**2]
                    and a[i] < a[(i+dim-2)%dim**2]
                    and a[i] < a[(i+dim+2)%dim**2]
                    and a[i] < a[(i+2*dim-1)%dim**2]
                    and a[i] < a[(i+2*dim+1)%dim**2]):
                    if(a[i] < a[(i-3*dim-1)%dim**2]
                        and a[i] < a[(i-3*dim+1)%dim**2]
                        and a[i] < a[(i-dim-3)%dim**2]
                        and a[i] < a[(i-dim+3)%dim**2]
                        and a[i] < a[(i+dim-3)%dim**2]
                        and a[i] < a[(i+dim+3)%dim**2]
                        and a[i] < a[(i+3*dim-1)%dim**2]
                        and a[i] < a[(i+3*dim+1)%dim**2]):
                        x_mins.append(x[i])
                        y_mins.append(y[i])
                        values.append(a[i])
    return np.array(x_mins), np.array(y_mins), np.array(values)


def half_light_radius(lens_light, x_grid, y_grid, center_x=0, center_y=0):
    """

    :param lens_light: array of surface brightness
    :param x_grid: x-axis coordinates
    :param y_gird: y-axis coordinates
    :param center_x: center of light
    :param center_y: center of light
    :return:
    """
    lens_light[lens_light < 0] = 0
    total_flux_2 = np.sum(lens_light)/2.
    lens_light_img = array2image(lens_light)
    r_max = np.max(np.sqrt(x_grid**2 + y_grid**2))
    for i in range(1000):
        r = i/500. * r_max
        mask = 1. - get_mask(center_x, center_y, r, x_grid, y_grid)
        flux_enclosed = np.sum(np.array(lens_light_img)*mask)
        if flux_enclosed > total_flux_2:
            return r
    return -1


def fwhm_kernel(kernel):
    """
    computes the full width at half maximum of a (PSF) kernel
    :param kernel: (psf) kernel, 2d numpy array
    :return: fwhm in units of pixels
    """
    n = len(kernel)
    if n % 2 == 0:
        raise ValueError('only works with odd number of pixels in kernel!')
    max_flux = kernel[(n-1)/2, (n-1)/2]
    I_2 = max_flux/2.
    I_r = kernel[(n-1)/2, (n-1)/2:]
    r = np.linspace(0, (n-1)/2, (n+1)/2)
    for i in range(1, len(r)):
        if I_r[i] < I_2:
            fwhm_2 = (I_2 - I_r[i-1])/(I_r[i] - I_r[i-1]) + r[i-1]
            return fwhm_2 * 2
    raise ValueError('The kernel did not drop to half the max value - fwhm not determined!')


def radial_profile(light_grid, x_grid, y_grid, center_x=0, center_y=0, n=None):
    """

    :param light_grid: array of surface brightness
    :param x_grid: x-axis coordinates
    :param y_gird: y-axis coordinates
    :param center_x: center of light
    :param center_y: center of light
    :param n: number of discrete steps
    :return:
    """
    light_img = array2image(light_grid)
    r_max = np.max(np.sqrt(x_grid**2 + y_grid**2))
    if n is None:
        n = int(np.sqrt(len(x_grid)))
    I_r = np.zeros(n)
    I_enclosed = 0
    r = np.linspace(1./n*r_max, r_max, n)
    for i, r_i in enumerate(r):
        mask = 1. - get_mask(center_x, center_y, r_i, x_grid, y_grid)
        flux_enclosed = np.sum(np.array(light_img)*mask)
        I_r[i] = flux_enclosed - I_enclosed
        I_enclosed = flux_enclosed
    return I_r, r


def re_size_array(x_in, y_in, input_values, x_out, y_out):
    """
    resizes 2d array (i.e. image) to new coordinates. So far only works with square output aligned with coordinate axis.
    :param x_in:
    :param y_in:
    :param input_values:
    :param x_out:
    :param y_out:
    :return:
    """
    interp_2d = scipy.interpolate.interp2d(x_in, y_in, input_values, kind='linear')
    #interp_2d = scipy.interpolate.RectBivariateSpline(x_in, y_in, input_values, kx=1, ky=1)
    out_values = interp_2d.__call__(x_out, y_out)
    return out_values


def fwhm2sigma(fwhm):
    """

    :param fwhm: full-widt-half-max value
    :return: gaussian sigma (sqrt(var))
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma


def sigma2fwhm(sigma):
    """

    :param sigma:
    :return:
    """
    fwhm = sigma * (2 * np.sqrt(2 * np.log(2)))
    return fwhm


def hyper2F2_array(a, b, c, d, x):
    """

    :param a:
    :param b:
    :param c:
    :param d:
    :param x:
    :return:
    """
    if isinstance(x, int) or isinstance(x, float):
        out = mpmath.hyp2f2(a, b, c, d, x)
    else:
        n = len(x)
        out = np.zeros(n)
        for i in range(n):
            out[i] = mpmath.hyp2f2(a, b, c, d, x[i])
    return out


def make_subgrid(ra_coord, dec_coord, subgrid_res=2):
    """
    return a grid with subgrid resolution
    :param ra_coord:
    :param dec_coord:
    :param subgrid_res:
    :return:
    """
    ra_array = array2image(ra_coord)
    dec_array = array2image(dec_coord)
    n = len(ra_array)
    d_ra_x = ra_array[0][1] - ra_array[0][0]
    d_ra_y = ra_array[1][0] - ra_array[0][0]
    d_dec_x = dec_array[0][1] - dec_array[0][0]
    d_dec_y = dec_array[1][0] - dec_array[0][0]

    ra_array_new = np.zeros((n*subgrid_res, n*subgrid_res))
    dec_array_new = np.zeros((n*subgrid_res, n*subgrid_res))
    for i in range(0, subgrid_res):
        for j in range(0, subgrid_res):
            ra_array_new[i::subgrid_res, j::subgrid_res] = ra_array + d_ra_x * (-1/2. + 1/(2.*subgrid_res) + j/float(subgrid_res)) + d_ra_y * (-1/2. + 1/(2.*subgrid_res) + i/float(subgrid_res))
            dec_array_new[i::subgrid_res, j::subgrid_res] = dec_array + d_dec_x * (-1/2. + 1/(2.*subgrid_res) + j/float(subgrid_res)) + d_dec_y * (-1/2. + 1/(2.*subgrid_res) + i/float(subgrid_res))

    ra_coords_sub = image2array(ra_array_new)
    dec_coords_sub = image2array(dec_array_new)
    return ra_coords_sub, dec_coords_sub


def re_size_grid(grid, numPix):
    """
    smooths a given grid to larger pixels
    """
    numGrid = len(grid)

    if numGrid == numPix: #if the grid has the same size as the pixelized image
        return grid
    else:
        numAverage = numGrid/numPix
        if int(numAverage) == numAverage:
            return averaging(grid, numGrid, numPix)
        else:
            raise ValueError("grid size = %f is not a integer factor of pixel size = %f " % (numGrid, numPix))


def re_size(image, factor=1):
    """
    resizes image with nx x ny to nx/factor x ny/factor
    :param image: 2d image with shape (nx,ny)
    :param factor: integer >=1
    :return:
    """
    if factor < 1:
        raise ValueError('scaling factor in re-sizing %s < 1' %factor)
    f = int(factor)
    nx, ny = np.shape(image)
    if int(nx/f) == nx/f and int(ny/f) == ny/f:
        small = image.reshape([nx/f, f, ny/f, f]).mean(3).mean(1)
        return small
    else:
        raise ValueError("scaling with factor %s is not possible with grid size %s, %s" %(f, nx, ny))


def cut_psf(psf_data, psf_size):
    """
    cut the psf properly
    :param psf_data: image of PSF
    :param psf_size: size of psf
    :return: re-sized and re-normalized PSF
    """
    kernel = cut_edges(psf_data, psf_size)
    kernel = kernel_norm(kernel)
    return kernel


def symmetry_average(image, symmetry):
    """
    symmetry averaged image
    :param image:
    :param symmetry:
    :return:
    """
    img_sym = np.zeros_like(image)
    angle = 360./symmetry
    for i in range(symmetry):
        img_sym += rotateImage(image, angle*i)
    img_sym /= symmetry
    return img_sym


class FFTConvolve(object):
    """
    fft convolution routines optimized for different scipy versions
    """

    def fftconvolve(self, in1, in2, int2_fft, mode="same"):
        """

        :param in1:
        :param in2:
        :param int2_fft:
        :param mode:
        :return:
        """
        if scipy.__version__ == '0.14.0':
            return self._fftconvolve_14(in1, in2, int2_fft, mode)
        else:
            return self._fftconvolve_18(in1, in2, int2_fft, mode)

    # scipy-0.18.0 compatible
    def _fftconvolve_18(self, in1, in2, int2_fft, mode="same"):
        """
        scipy routine scipy.signal.fftconvolve with kernel already fourier transformed
        """
        in1 = signaltools.asarray(in1)
        in2 = signaltools.asarray(in2)

        if in1.ndim == in2.ndim == 0:  # scalar inputs
            return in1 * in2
        elif not in1.ndim == in2.ndim:
            raise ValueError("in1 and in2 should have the same dimensionality")
        elif in1.size == 0 or in2.size == 0:  # empty arrays
            return signaltools.array([])

        s1 = signaltools.array(in1.shape)
        s2 = signaltools.array(in2.shape)

        shape = s1 + s2 - 1

        # Check that input sizes are compatible with 'valid' mode
        if signaltools._inputs_swap_needed(mode, s1, s2):
            # Convolution is commutative; order doesn't have any effect on output
            in1, s1, in2, s2 = in2, s2, in1, s1

        # Speed up FFT by padding to optimal size for FFTPACK
        fshape = [signaltools.fftpack.helper.next_fast_len(int(d)) for d in shape]
        fslice = tuple([slice(0, int(sz)) for sz in shape])
        # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
        # sure we only call rfftn/irfftn from one thread at a time.

        ret = np.fft.irfftn(np.fft.rfftn(in1, fshape) *
                    int2_fft, fshape)[fslice].copy()
        #np.fft.rfftn(in2, fshape)


        if mode == "full":
            return ret
        elif mode == "same":
            return signaltools._centered(ret, s1)
        elif mode == "valid":
            return signaltools._centered(ret, s1 - s2 + 1)
        else:
            raise ValueError("Acceptable mode flags are 'valid',"
                             " 'same', or 'full'.")


    # scipy-0.14.0 compatible
    def _fftconvolve_14(self, in1, in2, int2_fft, mode="same"):
        """
        scipy routine scipy.signal.fftconvolve with kernel already fourier transformed
        """
        in1 = signaltools.asarray(in1)
        in2 = signaltools.asarray(in2)

        if in1.ndim == in2.ndim == 0:  # scalar inputs
            return in1 * in2
        elif not in1.ndim == in2.ndim:
            raise ValueError("in1 and in2 should have the same dimensionality")
        elif in1.size == 0 or in2.size == 0:  # empty arrays
            return signaltools.array([])

        s1 = signaltools.array(in1.shape)
        s2 = signaltools.array(in2.shape)

        shape = s1 + s2 - 1

        # Speed up FFT by padding to optimal size for FFTPACK
        fshape = [signaltools._next_regular(int(d)) for d in shape]
        fslice = tuple([slice(0, int(sz)) for sz in shape])
        # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
        # sure we only call rfftn/irfftn from one thread at a time.

        ret = signaltools.irfftn(signaltools.rfftn(in1, fshape) *
                    int2_fft, fshape)[fslice].copy()
        #np.fft.rfftn(in2, fshape)


        if mode == "full":
            return ret
        elif mode == "same":
            return signaltools._centered(ret, s1)
        elif mode == "valid":
            return signaltools._centered(ret, s1 - s2 + 1)
        else:
            raise ValueError("Acceptable mode flags are 'valid',"
                             " 'same', or 'full'.")

    # scipy-0.18.0 compatible
    def _fftn_18(self, image, kernel):
        """
        return the fourier transpose of the kernel in same modes as image
        :param image:
        :param kernel:
        :return:
        """
        in1 = signaltools.asarray(image)
        in2 = signaltools.asarray(kernel)

        s1 = signaltools.array(in1.shape)
        s2 = signaltools.array(in2.shape)

        shape = s1 + s2 - 1

        fshape = [signaltools.fftpack.helper.next_fast_len(int(d)) for d in shape]
        kernel_fft = np.fft.rfftn(in2, fshape)
        return kernel_fft

    # scipy-0.14.0 compatible
    def _fftn_14(self, image, kernel):
        """
        return the fourier transpose of the kernel in same modes as image
        :param image:
        :param kernel:
        :return:
        """
        in1 = signaltools.asarray(image)
        in2 = signaltools.asarray(kernel)

        s1 = signaltools.array(in1.shape)
        s2 = signaltools.array(in2.shape)

        shape = s1 + s2 - 1

        fshape = [signaltools._next_regular(int(d)) for d in shape]
        kernel_fft = signaltools.rfftn(in2, fshape)
        return kernel_fft

    def fftn(self, image, kernel):
        """
        return the fourier transpose of the kernel in same modes as image
        :param image:
        :param kernel:
        :return:
        """
        if scipy.__version__ == '0.14.0':
            return self._fftn_14(image, kernel)
        else:
            return self._fftn_18(image, kernel)