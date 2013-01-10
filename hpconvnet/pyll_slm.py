"""
A library file for the design approach of cifar10.py

It includes the slm components as well as a few other things that should
migrate upstream.

"""
import cPickle
import logging
import time

import numpy as np
from skimage.transform import AffineTransform
from skimage.transform._warps_cy import _warp_fast

import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv

from hyperopt import pyll
import hyperopt

from skdata import larray

from .utils import mean_and_std
from .utils import dot_f32
from .utils import dot_f64
import foobar

import isvm_boosting # for worth_calculating


logger = logging.getLogger(__name__)


pyll.scope.define_info(o_len=2)(mean_and_std)


class InvalidDescription(Exception):
    """Model description was invalid"""


class EvalTimeout(Exception):
    """Document could not be evaluated in time
    """


global_timeout = float('inf')


def set_timeout(t):
    global global_timeout
    global_timeout = t


def raise_on_timeout():
    if time.time() > global_timeout:
        raise EvalTimeout()



class AffineImageWarp(theano.Op):

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash((type(self),))

    def make_node(self, x, rot, shear, scale, trans, oshape):
        inputs = map(theano.tensor.as_tensor_variable,
                [x, rot, shear, scale, trans, oshape])
        if 'float' not in x.dtype:
            raise TypeError('warping int images is not supported')
        return theano.Apply(self, inputs, [x.type()])

    def perform(self, node, inputs, out_storage):
        x, rot, shear, scale, trans, oshape = inputs
        #foobar.append_ndarray_signature(x, 'AffineImageWarp x')

        aff = AffineTransform(rotation=rot, shear=shear, scale=scale,
                translation=trans)

        if str(x.dtype) != node.inputs[0].dtype:
            raise TypeError("Wrong dtype argument to AffineImageWarp", x.dtype)

        if np.any(x < 0):
            raise ValueError('X should be positive')

        if np.any(x > 1.0):
            raise ValueError('X should be less than 1')

        N, C, H, W = x.shape
        rows, cols = oshape

        rval = out_storage[0][0]
        rval_shape = (N, C, rows, cols)

        if ((rval is None)
                or (rval.dtype != x.dtype)
                or rval.shape != rval_shape):
            rval3 = np.empty((N * C, rows, cols), dtype=x.dtype)
            bg_check = True
        else:
            rval3 = rval.reshape((N * C, rows, cols))
            bg_check = False

        xx = x.reshape(N * C, H, W)

        # -- a small exactly-representable float for out-of-bounds pixels
        oob = -1.0 / 2 ** 16
        order = 1  # TODO: TRY ORDER=2 (WHY DOES RANGE GET LARGER?)

        tform = np.linalg.inv(aff._matrix)

        for i in xrange(N * C):
            if bg_check and i == 0:
                rval3[i] = _warp_fast(xx[i], tform,
                        output_shape=oshape, order=order,
                        cval=oob)
                oob_ratio = np.mean(rval3[i] == oob)
                if oob_ratio > 0.5:
                    raise InvalidDescription('too much background', oob_ratio)
                rval3[i] = np.maximum(0, rval3[i])
            else:
                rval3[i] = _warp_fast(xx[i], np.linalg.inv(aff._matrix),
                        output_shape=oshape, order=order,
                        cval=0)

            if 0 and i == 0:
                print 'Debugprint from AffineImageWarp...'
                for sym in 'rot', 'shear', 'scale', 'trans', 'oshape':
                    print sym, ':', locals()[sym]
                import matplotlib.pyplot as pl
                pl.subplot(2, 1, 1)
                pl.imshow(xx[i], cmap=pl.cm.gray)
                pl.subplot(2, 1, 2)
                pl.imshow(rval3[i], cmap=pl.cm.gray)
                pl.show()
                time.sleep(2)  # -- give some time to ctrl-C

        if np.any(rval3 > 1.001) or np.any(rval3 < 0.0):
            min3 = np.min(rval3)
            max3 = np.max(rval3)
            raise ValueError('interpolated pixel values out of range',
                             (min3, max3))

        out_storage[0][0] = rval3.reshape(rval_shape)
        #foobar.append_ndarray_signature(out_storage[0][0], 'AffineImageWarp y')


affine_image_warp = AffineImageWarp()


@pyll.scope.define
def slm_affine_image_warp((x, x_shp),
        rot, shear, scale, trans, oshape):
    assert x_shp[2] == x_shp[3]
    z = affine_image_warp(x,
        rot, shear, np.asarray(scale), np.asarray(trans), np.asarray(oshape))
    z_shp = (x_shp[0], x_shp[1]) + tuple(oshape)
    assert z_shp[2] == z_shp[3]
    return z, z_shp


@pyll.scope.define
def slm_img_uint8_to_float32((x, x_shp),):
    if str(x.dtype) != 'uint8':
        raise TypeError('x must be uint8', x.dtype)
    return (x.astype('float32') / 255, x_shp)


@pyll.scope.define
def alloc_random_uniform_filterbank(n_filters, height, width,
        channels, dtype, rseed, normalize=True):
    """
    Generate the same weights as are generated by pythor3
    """
    if height != width:
        raise ValueError('filters must be square')
    if channels is None:
        filter_shape = [n_filters, height, width]
    else:
        filter_shape = [n_filters, height, width, channels]

    rng = np.random.RandomState(rseed)
    foobar.append_randomstate('alloc_random_uniform_filterbank', rng)
    fb_data = rng.uniform(size=filter_shape)

    # normalize each filter in the bank if needed
    if normalize:
        # TODO: vectorize these computations, do all at once.
        for fidx, filt in enumerate(fb_data):
            # normalization here means zero-mean, unit-L2norm
            filt -= filt.mean()
            filt_norm = np.sqrt((filt * filt).sum())
            assert filt_norm != 0
            filt /= filt_norm
            fb_data[fidx] = filt

    foobar.append_ndarray_signature(fb_data, 'alloc_random_uniform_filterbank')
    return fb_data.astype(dtype)


@pyll.scope.define_info(o_len=2)
def boxconv((x, x_shp), kershp, channels=False):
    """
    channels: sum over channels (T/F)
    """
    kershp = tuple(kershp)
    if channels:
        rshp = (   x_shp[0],
                    1,
                    x_shp[2] - kershp[0] + 1,
                    x_shp[3] - kershp[1] + 1)
        kerns = np.ones((1, x_shp[1]) + kershp, dtype=x.dtype)
    else:
        rshp = (   x_shp[0],
                    x_shp[1],
                    x_shp[2] - kershp[0] + 1,
                    x_shp[3] - kershp[1] + 1)
        kerns = np.ones((1, 1) + kershp, dtype=x.dtype)
        x_shp = (x_shp[0]*x_shp[1], 1, x_shp[2], x_shp[3])
        x = x.reshape(x_shp)
    try:
        rval = tensor.reshape(
                conv.conv2d(x,
                    theano.shared(kerns),
                    image_shape=x_shp,
                    filter_shape=kerns.shape,
                    border_mode='valid'),
                rshp)
    except Exception, e:
        if "Bad size for the output shape" in str(e):
            raise InvalidDescription('boxconv', (x_shp, kershp, channels))
        else:
            raise
    return rval, rshp


@pyll.scope.define_info(o_len=2)
def slm_fbcorr_chmaj((x, x_shp), kerns, stride=1, mode='valid'):
    """
    Channel-major filterbank correlation

    kerns - filterbank with shape (n_filters, ker_size, ker_size, channels)

    """
    assert x_shp[2] == x_shp[3]
    assert x.dtype == 'float32'
    # Reference implementation:
    # ../pythor3/pythor3/operation/fbcorr_/plugins/scipy_naive/scipy_naive.py
    if stride != 1:
        raise NotImplementedError('stride is not used in reference impl.')

    # -- flip the kernels so that convolution does correlation
    kerns = kerns[:, :, ::-1, ::-1]
    s_kerns = theano.shared(kerns.astype(x.dtype))
    x = conv.conv2d(
            x,
            s_kerns,
            image_shape=x_shp,
            filter_shape=kerns.shape,
            border_mode=mode)

    n_filters, channels, krows, kcols = kerns.shape
    if mode == 'valid':
        x_shp = (x_shp[0], n_filters,
                x_shp[2] - krows + 1,
                x_shp[3] - kcols + 1)
    elif mode == 'full':
        x_shp = (x_shp[0], n_filters,
                x_shp[2] + krows - 1,
                x_shp[3] + kcols - 1)
    else:
        raise NotImplementedError('fbcorr mode', mode)

    assert x_shp[2] == x_shp[3]
    return x, x_shp


@pyll.scope.define_info(o_len=2)
def slm_clipout((x, x_shp), min_out, max_out):
    if min_out is None and max_out is None:
        return x, x_shp
    elif min_out is None:
        return tensor.minimum(x, max_out), x_shp
    elif max_out is None:
        return tensor.maximum(x, min_out), x_shp
    else:
        return tensor.clip(x, min_out, max_out), x_shp


@pyll.scope.define_info(o_len=2)
def slm_lpool((x, x_shp),
        ker_size=3,
        order=1,
        stride=1,
        mode='valid'):
    assert x.dtype == 'float32'
    assert x_shp[2] == x_shp[3]
    order=float(order)

    ker_shape = (ker_size, ker_size)
    if hasattr(order, '__iter__'):
        o1 = (order == 1).all()
        o2 = (order == order.astype(np.int)).all()
    else:
        o1 = order == 1
        o2 = (order == int(order))

    if o1:
        r, r_shp = boxconv((x, x_shp), ker_shape)
    elif o2:
        r, r_shp = boxconv((x ** order, x_shp), ker_shape)
        r = tensor.maximum(r, 0) ** (1.0 / order)
    else:
        r, r_shp = boxconv((abs(x) ** order, x_shp), ker_shape)
        r = tensor.maximum(r, 0) ** (1.0 / order)

    if stride > 1:
        r = r[:, :, ::stride, ::stride]
        # intdiv is tricky... so just use numpy
        r_shp = np.empty(r_shp)[:, :, ::stride, ::stride].shape
    assert r_shp[2] == r_shp[3]
    return r, r_shp


@pyll.scope.define_info(o_len=2)
def slm_lnorm((x, x_shp),
        ker_size=3,
        remove_mean= False,
        div_method='euclidean',
        threshold=0.0,
        stretch=1.0,
        mode='valid',
        EPSILON=1e-4,
        ):
    # Reference implementation:
    # ../pythor3/pythor3/operation/lnorm_/plugins/scipy_naive/scipy_naive.py
    assert x.dtype == 'float32'
    assert x_shp[2] == x_shp[3]
    inker_shape=(ker_size, ker_size)
    outker_shape=(ker_size, ker_size)  # (3, 3)
    if mode != 'valid':
        raise NotImplementedError('lnorm requires mode=valid', mode)

    threshold = float(threshold)
    stretch = float(stretch)

    if outker_shape == inker_shape:
        size = np.asarray(x_shp[1] * inker_shape[0] * inker_shape[1],
                dtype=x.dtype)
        ssq, ssqshp = boxconv((x ** 2, x_shp), inker_shape,
                channels=True)
        xs = inker_shape[0] // 2
        ys = inker_shape[1] // 2
        # --local contrast normalization in regions that are not symmetric
        #   about the pixel being normalized feels weird, but we're
        #   allowing it here.
        xs_inc = (inker_shape[0] + 1) % 2
        ys_inc = (inker_shape[1] + 1) % 2
        if div_method == 'euclidean':
            if remove_mean:
                arr_sum, _shp = boxconv((x, x_shp), inker_shape,
                        channels=True)
                arr_num = (x[:, :, xs-xs_inc:-xs, ys-ys_inc:-ys]
                        - arr_sum / size)
                arr_div = EPSILON + tensor.sqrt(
                        tensor.maximum(0,
                            ssq - (arr_sum ** 2) / size))
            else:
                arr_num = x[:, :, xs-xs_inc:-xs, ys-ys_inc:-ys]
                arr_div = EPSILON + tensor.sqrt(ssq)
        else:
            raise NotImplementedError('div_method', div_method)
    else:
        raise NotImplementedError('outker_shape != inker_shape',
                outker_shape, inker_shape)

    if (hasattr(stretch, '__iter__') and (stretch != 1).any()) or stretch != 1:
        arr_num = arr_num * stretch
        arr_div = arr_div * stretch
    # XXX: IS THIS 1.0 supposed to be (threshold + EPSILON) ??
    arr_div = tensor.switch(arr_div < (threshold + EPSILON), 1.0, arr_div)

    r = arr_num / arr_div
    r_shp = x_shp[0], x_shp[1], ssqshp[2], ssqshp[3]
    return r, r_shp


@pyll.scope.define_info(o_len=2)
def slm_fbncc_chmaj((x, x_shp), m_fb, remove_mean, beta, hard_beta):
    """
    Channel-major filterbank normalized cross-correlation

    For each valid-mode patch (p) of the image (x), this transform computes

    p_c = (p - mean(p)) if (remove_mean) else (p)
    qA = p_c / sqrt(var(p_c) + beta)           # -- Coates' sc_vq_demo
    qB = p_c / sqrt(max(sum(p_c ** 2), beta))  # -- Pinto's lnorm

    There are two differences between qA and qB:

    1. the denominator contains either addition or max

    2. the denominator contains either var or sum of squares

    The first difference corresponds to the hard_beta parameter.
    The second difference amounts to a decision about the scaling of the
    output, because for every function qA(beta_A) there is a function
    qB(betaB) that is identical, except for a multiplicative factor of
    sqrt(N - 1).

    I think that in the context of stacked models, the factor of sqrt(N-1) is
    undesirable because we want the dynamic range of all outputs to be as
    similar as possible. So this function implements qB.

    Coates' denominator had var(p_c) + 10, so what should the equivalent here
    be?
    p_c / sqrt(var(p_c) + 10)
    = p_c / sqrt(sum(p_c ** 2) / (108 - 1) + 10)
    = p_c / sqrt((sum(p_c ** 2) + 107 * 10) / 107)
    = sqrt(107) * p_c / sqrt((sum(p_c ** 2) + 107 * 10))

    So Coates' pre-processing has beta = 1070, hard_beta=False. This function
    returns a result that is sqrt(107) ~= 10 times smaller than the Coates
    whitening step.

    """
    # -- just to make sure things will run on GPU
    assert x.dtype == 'float32'
    w_means, w_fb = m_fb

    beta = float(beta)

    # -- kernel Number, Features, Rows, Cols
    kN, kF, kR, kC = w_fb.shape

    # -- patch-wise sums and sums-of-squares
    p_sum, _shp = boxconv((x, x_shp), (kR, kC), channels=True)
    p_mean = 0 if remove_mean else p_sum / (kF * kR * kC)
    p_ssq, _shp = boxconv((x ** 2, x_shp), (kR, kC), channels=True)

    # -- this is an important variable in the math above, but
    #    it is not directly used in the fused lnorm_fbcorr
    # p_c = x[:, :, xs - xs_inc:-xs, ys - ys_inc:-ys] - p_mean

    # -- adjust the sum of squares to reflect remove_mean
    p_c_sq = p_ssq - (p_mean ** 2) * (kF * kR * kC)
    if hard_beta:
        p_div2 = tensor.maximum(p_c_sq, beta)
    else:
        p_div2 = p_c_sq + beta

    p_scale = 1.0 / tensor.sqrt(p_div2)

    # --
    # from whitening, we have a shift and linear transform (P)
    # for each patch (as vector).
    #
    # let m be the vector [m m m m] that replicates p_mean
    # let a be the scalar p_scale
    # let x be an image patch from s_imgs
    #
    # Whitening means applying the affine transformation
    #   (c - M) P
    # to contrast-normalized patch c = a (x - m),
    # where a = p_scale and m = p_mean.
    #
    # We also want to extract features in dictionary
    #
    #   (c - M) P
    #   = (a (x - [m,m,m]) - M) P
    #   = (a x - a [m,m,m] - M) P
    #   = a x P - a [m,m,m] P - M P
    #

    P = theano.shared(
            np.asarray(w_fb[:, :, ::-1, ::-1], order='C'))

    Px = conv.conv2d(x, P,
            image_shape=x_shp,
            filter_shape=w_fb.shape,
            border_mode='valid')

    s_P_sum = theano.shared(w_fb.sum(3).sum(2).sum(1))
    Pmmm = p_mean * s_P_sum.dimshuffle(0, 'x', 'x')
    s_PM = theano.shared((w_means * w_fb).sum(3).sum(2).sum(1))
    z = p_scale * (Px - Pmmm) - s_PM.dimshuffle(0, 'x', 'x')

    assert z.dtype == x.dtype, (z.dtype, x.dtype)
    return z, (_shp[0], kN, _shp[2], _shp[3])


@pyll.scope.define
def slm_flatten((x, x_shp),):
    r = tensor.flatten(x, 2)
    r_shp = x_shp[0], np.prod(x_shp[1:])
    return r, r_shp


@pyll.scope.define_info(o_len=2)
def slm_lpool_smallgrid((x, x_shp), grid_res=2, order=1):
    """
    Like lpool, but parametrized to produce a fixed size image as output.
    The image is not rescaled, but rather single giant box filters are
    defined for each output pixel, and stored in a matrix.
    """
    assert x.dtype == 'float32'
    order=float(order)

    if hasattr(order, '__iter__'):
        o1 = (order == 1).all()
        o2 = (order == order.astype(np.int)).all()
    else:
        o1 = order == 1
        o2 = (order == int(order))

    # rather than convolving with a box, this function takes
    # a dot product with the entire image
    ngR = x_shp[2] // grid_res + int(x_shp[2] % grid_res > 0)
    ngC = x_shp[3] // grid_res + int(x_shp[3] % grid_res > 0)

    assert ngR * grid_res >= x_shp[2]
    assert ngC * grid_res >= x_shp[3]

    W = np.zeros((grid_res, grid_res,) + x_shp[2:], dtype=x.dtype)
    for rr in range(grid_res):
        for cc in range(grid_res):
            W[rr, cc,
                    rr * ngR : (rr + 1) * ngR,
                    cc * ngC : (cc + 1) * ngC] = 1.0
    sW = theano.shared(W.reshape((grid_res ** 2, -1)))

    xmat = x.reshape((x_shp[0] * x_shp[1], x_shp[2] * x_shp[3]))

    if o1:
        r = tensor.dot(xmat, sW.T)
    elif o2:
        r = tensor.sqrt(tensor.dot(xmat ** 2, sW.T))
    else:
        r = tensor.dot(abs(xmat) ** order, sW.T)
        r = tensor.maximum(r, 0) ** (1.0 / order)

    r_shp = (x_shp[0], x_shp[1], grid_res, grid_res)
    r = r.reshape(r_shp)

    return r, r_shp


@pyll.scope.define_info(o_len=2)
def slm_quantize_gridpool((x, x_shp), alpha,
        use_mid=False,
        order=1.0,
        grid_res=2):
    hr = int(np.round(x_shp[2] / grid_res))
    hc = int(np.round(x_shp[3] / grid_res))
    alpha = tensor.cast(alpha, dtype=x.dtype)
    sXC_shp = (x_shp[0], x_shp[1], grid_res, grid_res, 3 if use_mid else 2)
    sXC = tensor.zeros(sXC_shp, dtype=x.dtype)

    for ri in range(grid_res):
        if ri == grid_res - 1:
            rslice = slice(ri * hr, None)
        else:
            rslice = slice(ri * hr, (ri + 1) * hr)
        for ci in range(grid_res):
            cslice = slice(ci * hc, (ci + 1) * hc)
            if ci == grid_res - 1:
                cslice = slice(ci * hc, None)
            else:
                cslice = slice(ci * hc, (ci + 1) * hc)
            xi = x[:, :, rslice, cslice]
            qs = []
            qs.append(tensor.maximum(xi - alpha, 0))
            qs.append(tensor.maximum(-xi - alpha, 0))
            if use_mid:
                qs.append(tensor.maximum(alpha - abs(xi), 0))

            for qi, q in enumerate(qs):
                inc = (q ** order).sum([2, 3]) ** (1. / order)
                assert inc.dtype == q.dtype
                sXC = tensor.set_subtensor(sXC[:, :, ri, ci, qi], inc)

    r_shp = sXC_shp[0], np.prod(sXC_shp[1:])
    r = sXC.reshape(r_shp)
    return r, r_shp


@pyll.scope.define_info(o_len=2)
def slm_lpool_alpha((x, x_shp),
        ker_size=3,
        order=1,
        stride=1,
        alpha=0.0,
        ):
    """
    lpool but with alpha-half-rectification
    """
    assert x.dtype == 'float32'
    order=float(order)

    ker_shape = (ker_size, ker_size)

    xp = tensor.maximum(x - alpha, 0)
    xn = tensor.maximum(-x - alpha, 0)
    rp, r_shp = boxconv((xp ** order, x_shp), ker_shape)
    rn, r_shp = boxconv((xn ** order, x_shp), ker_shape)
    rp = rp ** (1. / order)
    rn = rn ** (1. / order)

    if stride > 1:
        # -- theano optimizations should turn this stride into conv2d
        #    subsampling
        rp = rp[:, :, ::stride, ::stride]
        rn = rn[:, :, ::stride, ::stride]
        # intdiv is tricky... so just use numpy
        r_shp = np.empty(r_shp)[:, :, ::stride, ::stride].shape

    z_shp = (r_shp[0], 2 * r_shp[1], r_shp[2], r_shp[3])
    z = tensor.zeros(z_shp, dtype=x.dtype)
    z = tensor.set_subtensor(z[:, :r_shp[1]], rp)
    z = tensor.set_subtensor(z[:, r_shp[1]:], rn)

    return z, z_shp


@pyll.scope.define_info(o_len=2)
def slm_gnorm((x, x_shp),
        remove_mean= False,
        div_method='euclidean',
        threshold=0.0,
        stretch=1.0,
        EPSILON=1e-4,
        across_channels=True,
        ):
    """
    Global normalization, as opposed to local normalization
    """

    threshold = float(threshold)
    stretch = float(stretch)

    if across_channels:
        size = x_shp[1] * x_shp[2] * x_shp[3]
        ssq = (x ** 2).sum(axis=[1, 2, 3]).dimshuffle(0, 'x', 'x', 'x')
    else:
        size = x_shp[2] * x_shp[3]
        ssq = (x ** 2).sum(axis=[2, 3]).dimshuffle(0, 1, 'x', 'x')

    if div_method == 'euclidean':
        if remove_mean:
            if across_channels:
                arr_sum = x.sum(axis=[1, 2, 3]).dimshuffle(0, 'x', 'x', 'x')
            else:
                arr_sum = x.sum(axis=[2, 3]).dimshuffle(0, 1, 'x', 'x')

            arr_num = x - arr_sum / size
            arr_div = EPSILON + tensor.sqrt(
                    tensor.maximum(0,
                        ssq - (arr_sum ** 2) / size))
        else:
            arr_num = x
            arr_div = EPSILON + tensor.sqrt(ssq)
    else:
        raise NotImplementedError('div_method', div_method)

    if (hasattr(stretch, '__iter__') and (stretch != 1).any()) or stretch != 1:
        arr_num = arr_num * stretch
        arr_div = arr_div * stretch
    arr_div = tensor.switch(arr_div < (threshold + EPSILON), 1.0, arr_div)

    r = arr_num / arr_div
    r_shp = x_shp
    return r, r_shp


@pyll.scope.define
def contrast_normalize(patches, remove_mean, beta, hard_beta):
    X = patches
    if X.ndim != 2:
        raise TypeError('contrast_normalize requires flat patches')
    if remove_mean:
        xm = X.mean(1)
    else:
        xm = X[:,0] * 0
    Xc = X - xm[:, None]
    if 0:
        # -- for some reason the following sometimes uses gigs of RAM
        l2 = (Xc * Xc).sum(axis=1)
    else:
        l2 = np.zeros_like(Xc[:, 0])
        for i in xrange(Xc.shape[1]):
            l2 += Xc[:, i] ** 2
    if hard_beta:
        div2 = np.maximum(l2, beta)
    else:
        div2 = l2 + beta
    Xc /= np.sqrt(div2[:, None])
    foobar.append_ndarray_signature(Xc, 'contrast_normalize')
    return Xc


@pyll.scope.define
def random_patches(images, N, R, C, rng, channel_major=False, memlimit=None):
    """Return a stack of N image patches (channel major version)"""

    def N_with_memlimit():
        if memlimit is not None:
            # -- memlimit in bytes
            sizelimit = memlimit / images.dtype.itemsize
            return min(N, sizelimit // (R * C * iF))
        else:
            return N

    if channel_major:
        n_imgs, iF, iR, iC = images.shape
        N = N_with_memlimit()
        rval = np.empty((N, iF, R, C), dtype=images.dtype)
    else:
        n_imgs, iR, iC, iF = images.shape
        N = N_with_memlimit()
        rval = np.empty((N, R, C, iF), dtype=images.dtype)

    foobar.append_trace('random_patches dims', *rval.shape)
    foobar.append_randomstate('random_patches rng', rng)

    srcs = rng.randint(n_imgs, size=N)

    if R > iR or C > iC:
        raise InvalidDescription('cannot extract patches', (R, C))
    roffsets = rng.randint(iR - R + 1, size=N)
    coffsets = rng.randint(iC - C + 1, size=N)
    # TODO: this can be done with one advanced index right?
    for rv_i, src_i, ro, co in zip(rval, srcs, roffsets, coffsets):
        if channel_major:
            rv_i[:] = images[src_i, :, ro: ro + R, co : co + C]
        else:
            rv_i[:] = images[src_i, ro: ro + R, co : co + C]
    foobar.append_ndarray_signature(rval, 'random_patches rval')
    return rval


@pyll.scope.define_info(o_len=3)
def patch_whitening_filterbank_X(patches, o_ndim, gamma,
        remove_mean, beta, hard_beta,
        ):
    """
    patches - Image patches (can be uint8 pixels or floats)
    o_ndim - 2 to get matrix outputs, 4 to get image-stack outputs
    gamma - non-negative real to boost low-principle components

    remove_mean - see contrast_normalize
    beta - see contrast_normalize
    hard_beta - see contrast_normalize

    Returns: M, P, X
        M - mean of contrast-normalized patches
        P - whitening matrix / filterbank for contrast-normalized patches
        X - contrast-normalized patches

    """
    # Algorithm from Coates' sc_vq_demo.m

    # -- patches -> column vectors
    X = patches.reshape(len(patches), -1).astype('float64')

    X = contrast_normalize(X,
            remove_mean=remove_mean,
            beta=beta,
            hard_beta=hard_beta)

    # -- ZCA whitening (with low-pass)
    logger.debug('patch_whitening_filterbank_X starting ZCA')
    M, _std = mean_and_std(X)
    Xm = X - M
    assert Xm.shape == X.shape
    logger.info('patch_whitening_filterbank_X starting ZCA: dot %s' %
            str(Xm.shape))
    C = dot_f64(Xm.T, Xm) / (Xm.shape[0] - 1)
    logger.debug('patch_whitening_filterbank_X starting ZCA: eigh')
    D, V = np.linalg.eigh(C)
    logger.debug(
        'patch_whitening_filterbank_X starting ZCA: dot %s' % str(V.shape))
    P = dot_f32(np.sqrt(1.0 / (D + gamma)) * V, V.T)

    # -- return to image space
    if o_ndim == 4:
        M = M.reshape(patches.shape[1:])
        P = P.reshape((P.shape[0],) + patches.shape[1:])
        X = X.reshape((len(X),) + patches.shape[1:])
    elif o_ndim == 2:
        pass
    else:
        raise ValueError('o_ndim not in (2, 4)', o_ndim)

    logger.debug('patch_whitening_filterbank_X -> done')

    foobar.append_ndarray_signature(M, 'patch_whitening_filterbank_X M')
    foobar.append_ndarray_signature(P, 'patch_whitening_filterbank_X P')
    foobar.append_ndarray_signature(X, 'patch_whitening_filterbank_X X')
    dtype = patches.dtype
    return M.astype(dtype), P.astype(dtype), X.astype(dtype)


@pyll.scope.define_info(o_len=2)
def fb_whitened_projections(patches, pwfX, n_filters, rseed, dtype):
    """
    pwfX is the output of patch_whitening_filterbank_X with reshape=False

    M, and fb will be reshaped to match elements of patches
    """
    M, P, patches_cn = pwfX
    if patches_cn.ndim != 2:
        raise TypeError('wrong shape for pwfX args, should be flattened',
                patches_cn.shape)
    rng = np.random.RandomState(rseed)
    foobar.append_randomstate('fb_whitened_projections', rng)

    D = rng.randn(n_filters, patches_cn.shape[1])
    D = D / (np.sqrt((D ** 2).sum(axis=1))[:, None] + 1e-20)
    fb = dot_f32(D, P)
    fb.shape = (n_filters,) + patches.shape[1:]
    M.shape = patches.shape[1:]
    M = M.astype(dtype)
    fb = fb.astype(dtype)
    if fb.size == 0:
        raise ValueError('filterbank had size 0')
    foobar.append_ndarray_signature(M, 'fb_whitened_projections M')
    foobar.append_ndarray_signature(fb, 'fb_whitened_projections fb')
    return M, fb


@pyll.scope.define_info(o_len=2)
def fb_whitened_patches(patches, pwfX, n_filters, rseed, dtype):
    """
    pwfX is the output of patch_whitening_filterbank_X with reshape=False

    M, and fb will be reshaped to match elements of patches

    """
    M, P, patches_cn = pwfX
    rng = np.random.RandomState(rseed)
    foobar.append_randomstate('fb_whitened_patches', rng)
    d_elems = rng.randint(len(patches_cn), size=n_filters)
    D = dot_f64(patches_cn[d_elems] - M, P)
    D = D / (np.sqrt((D ** 2).sum(axis=1))[:, None] + 1e-20)
    fb = dot_f32(D, P)
    fb.shape = (n_filters,) + patches.shape[1:]
    M.shape = patches.shape[1:]
    M = M.astype(dtype)
    fb = fb.astype(dtype)
    if fb.size == 0:
        raise ValueError('filterbank had size 0')
    foobar.append_ndarray_signature(M, 'fb_whitened_patches M')
    foobar.append_ndarray_signature(fb, 'fb_whitened_patches fb')
    return M, fb


@pyll.scope.define
def pyll_theano_batched_lmap(pipeline, seq, batchsize,
        _debug_call_counts=None,
        print_progress_every=float('inf'),
        abort_on_rows_larger_than=None,
        speed_thresh=None,
        x_dtype='float32',
        ):
    """
    This function returns a skdata.larray.lmap object whose function
    is defined by a theano expression.

    The theano expression will be built and compiled specifically for the
    dimensions of the given `seq`. Therefore, in_rows, and out_rows should
    actually be a *pyll* graph, that evaluates to a theano graph.
    """

    in_shp = (batchsize,) + seq.shape[1:]
    batch = np.zeros(in_shp, dtype=x_dtype)
    s_ibatch = theano.shared(batch)
    s_xi = theano.tensor.as_tensor_variable(s_ibatch).type()
    s_N = s_xi.shape[0]
    s_X = theano.tensor.set_subtensor(s_ibatch[:s_N], s_xi)
    #print 'PIPELINE', pipeline
    thing = pipeline((s_X, in_shp))
    #print 'THING'
    #print thing
    #print '==='
    s_obatch, oshp = pyll.rec_eval(thing)
    assert oshp[0] == batchsize
    logger.info('batched_lmap oshp %s' % str(oshp))
    if abort_on_rows_larger_than:
        rowlen = np.prod(oshp[1:])
        if rowlen > abort_on_rows_larger_than:
            raise ValueError('rowlen %i exceeds limit %i' % (
                rowlen, abort_on_rows_larger_than))

    # Compile a function that takes a variable number of elements in,
    # returns the same number of processed elements out,
    # but does all internal computations using a fixed number of elements,
    # because convolutions are fastest when they're hard-coded to a certain
    # size.
    logger.debug('pyll_theano_batched_lmap compiling fn')
    _fn = theano.function([theano.Param(s_xi, strict=True)],
            s_obatch[:s_N],
            updates={
                s_ibatch: s_X, # this allows the inc_subtensor to be in-place
                })
    logger.debug('pyll_theano_batched_lmap compiling fn -> done')

    sums = {'elems': 0, 'times': 0.0}
    if speed_thresh is None:
        time_fn = _fn
    else:
        def time_fn(X):
            t0 = time.time()
            if str(X.dtype) != x_dtype:
                print 'time_fn dtype problem', X.dtype, x_dtype
            rval = _fn(X)
            dt = time.time() - t0
            #print 'DEBUG time_fn dt:', dt
            sums['elems'] += len(X)
            sums['times'] += dt
            return rval

        def raise_if_slow():
            exc = EvalTimeout(
                'batched_lmap failed to compute %i elements in %f secs'
                % (speed_thresh['elements'], speed_thresh['seconds']))
            if sums['elems'] >= speed_thresh['elements']:
                observed_ratio = sums['elems'] / sums['times']
                required_ratio = (speed_thresh['elements'] /
                        speed_thresh['seconds'])
                if observed_ratio < required_ratio:
                    raise exc
                else:
                    sums['elems'] = 0
                    sums['times'] = 0.0

    def fn_1(x):
        if _debug_call_counts:
            _debug_call_counts['fn_1'] += 1
        return time_fn(x[None, :, :, :])[0]

    attrs = {
            'shape': oshp[1:],
            'ndim': len(oshp) -1,
            'dtype': s_obatch.dtype }
    def rval_getattr(attr, objs):
        # -- objs don't matter to the structure of the return value
        try:
            return attrs[attr]
        except KeyError:
            raise AttributeError(attr)

    fn_1.rval_getattr = rval_getattr

    last_print_time = [time.time()]

    def check_for_print(offset, X):
        curtime = time.time()
        if (curtime - last_print_time[0]) > print_progress_every:
            logger.info('pyll_theano_batched_lmap.f_map %i %i' % (
                offset, len(X)))
            last_print_time[0] = curtime

        if speed_thresh is not None:
            raise_if_slow()

    def f_map(X):
        if _debug_call_counts:
            _debug_call_counts['f_map'] += 1

        if len(X) == batchsize:
            check_for_print(offset=0, X=X)
            return time_fn(X)

        rval = np.empty((len(X),) + oshp[1:], dtype=s_obatch.dtype)
        offset = 0
        while offset < len(X):
            check_for_print(offset, X)
            xi = X[offset: offset + batchsize]
            fn_i = time_fn(xi)
            if not np.all(np.isfinite(fn_i)):
                raise ValueError('non-finite features')
            rval[offset:offset + len(xi)] = fn_i
            offset += len(xi)
        return rval

    return larray.lmap(fn_1, seq, f_map=f_map)


@pyll.scope.define
def np_transpose(obj, arg):
    return obj.transpose(*arg)


@pyll.scope.define
def np_RandomState(rseed):
    rval = np.random.RandomState(rseed)
    return rval


@pyll.scope.define
def flatten_elems(obj):
    return obj.reshape(len(obj), -1)


@pyll.scope.define
def model_predict(mdl, X):
    return mdl.predict(X)


@pyll.scope.define
def model_decisions(mdl, X):
    return mdl.decisions(X)


@pyll.scope.define
def pickle_dumps(obj, protocol=None):
    if protocol is None:
        return cPickle.dumps(obj)
    else:
        return cPickle.dumps(obj, protocol=protocol)


@pyll.scope.define
def error_rate(pred, y):
    return np.mean(pred != y)


@pyll.scope.define
def print_ndarray_summary(msg, X):
    print msg, X.dtype, X.shape, X.min(), X.max(), X.mean()
    return X


@pyll.scope.define_info(o_len=2)
def slm_uniform_M_FB(nfilters, size, channels, rseed, normalize, dtype,
        ret_cmajor):
    print 'Allocating uniform filterbank', nfilters, size, channels
    M = np.asarray(0).reshape((1, 1, 1)).astype(dtype)
    FB = alloc_random_uniform_filterbank(
                    nfilters, size, size, channels,
                    dtype=dtype,
                    rseed=rseed,
                    normalize=normalize)
    if FB.size == 0:
        raise ValueError('filterbank had size 0')
    if ret_cmajor:
        return M, FB.transpose(0, 3, 1, 2)
    else:
        return M, FB


@pyll.scope.define
def larray_cache_memory(obj):
    return larray.cache_memory(obj)


@pyll.scope.define
def larray_cache_memmap(obj, name, basedir=None, msg=None):
    return larray.cache_memmap(obj, name, basedir=basedir, msg=msg)


@pyll.scope.define
def ceildiv(a, b):
    return int(np.ceil(float(a) / float(b)))


@pyll.scope.define
def view2_worth_calculating(loss, ctrl, thresh_loss, thresh_rank):
    #
    #  Decide whether to bother calculating the view2 score, which is slow.
    #

    if thresh_loss is not None and loss > thresh_loss:
        logger.info('worth_calculating_view2: False (loss %f > thresh %f)' % (
            loss, thresh_loss))
        return False
    elif ctrl is None:
        logger.info('worth_calculating_view2: True (ctrl is None)')
        return True
    else:
        trials = ctrl.trials
        # -- old logic
        if hasattr(trials, 'handle'):
            # -- hack for mongodb
            query = {
                    'result.status': hyperopt.STATUS_OK,
                    'exp_key': trials._exp_key,
                    }
            #docs = list(trials.handle.jobs.find(query, {'result.loss': 1}))
            docs = isvm_boosting.BoostHelper.query_MongoTrials(trials,
                    query={
                        'result.status': hyperopt.STATUS_OK,
                        #'misc.boosting.continues': {'$in': [None, parent_tid]},
                        })
        else:
            # -- impl for local trials object
            trials.refresh()
            docs = [d for d in trials.trials
                    if (
                        'result' in d
                        and d['result']['status'] == hyperopt.STATUS_OK
                        and d['exp_key'] == trials._exp_key
                        )]
        bh = isvm_boosting.BoostHelper(docs)
        cur_parent = bh.continues(ctrl.current_trial)
        cur_parent_tid = cur_parent['tid'] if cur_parent else None
        best_sibling = bh.best_child(cur_parent)
        if best_sibling:
            logger.info(
                    'view2_worth_calculating cur_parent:%s best_sibling:%s(%s)'
                    % (cur_parent_tid, best_sibling['tid'],
                        best_sibling['result']['loss']))
            if loss <= best_sibling['result']['loss']:
                return True
            else:
                if thresh_rank > 1:
                    raise NotImplementedError('thresh_rank')
                return False
        else:
            logger.info(
                    'view2_worth_calculating cur_parent:%s, best_child:None'
                    % cur_parent_tid)
            return True
        #losses = [d['result']['loss'] for d in docs]
        #losses.sort()
        #if len(losses) < thresh_rank:
            #logger.info('worth_calculating_view2: True (small len(losses))')
            #return True
        #else:
            #rank = np.searchsorted(losses, loss)
            #rval = rank < thresh_rank
            #logger.info('worth_calculating_view2: %s (rank %i / %i)' % (
                #rval, rank, len(losses)))
            #return rval


@pyll.scope.define
def average_row_l2norm(X):
    return np.sqrt((np.asarray(X) ** 2).sum(axis=1)).mean()

