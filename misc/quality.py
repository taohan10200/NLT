
import numpy
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
import numpy as np
import math
"""
Hat tip: http://stackoverflow.com/a/5078155/1828289
"""
from skimage.measure import compare_psnr, compare_ssim

def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (int(A.shape[0]/ block[0]), int(A.shape[1]/ block[1]))+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)


def get_ssim(img1, img2, C1=0.01**2, C2=0.03**2):

    bimg1 = block_view(img1, (11,11))
    bimg2 = block_view(img2, (11,11))
    s1  = numpy.sum(bimg1, (-1, -2))
    s2  = numpy.sum(bimg2, (-1, -2))
    ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
    s12 = numpy.sum(bimg1*bimg2, (-1, -2))

    vari = ss - s1*s1 - s2*s2
    covar = s12 - s1*s2

    ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))

    return numpy.mean(ssim_map)



# def psnr(img1, img2):
#     mse = numpy.mean( (img1 - img2) ** 2 )
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def get_psnr(im_true, im_test):
    pixel_max = np.max(im_true)
    if pixel_max != 0:
        im_true = im_true/pixel_max*255
    pixel_max = np.max(im_test)
    if pixel_max != 0:
        im_test = im_test/pixel_max*255

    return compare_psnr(im_true,im_test,data_range=255)