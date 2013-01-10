"""
This file supports the incremental construction of an SVM classifier
by partially-corrective boosting on the hinge loss.


Each incremental solver minimizes

    hinge( dot(X, w) + b + alpha * prev_Wx)
    + lambda * (|w|^2 + |alpha * prev_W|^2)


Each solver is designed to be run on a subset of all available features.

"""
import copy
import logging
import gc
import os
import shutil
import sys

import numpy as np
import autodiff
import theano # abstraction leak to pass mode for optimization

import tempfile
import skdata.larray # for large tempfile creation (tempdir isn't always big enough)

from .utils import dot

logger = logging.getLogger(__name__)
info = logger.info
warn = logger.warn
error = logger.error

_default_bfgs_kwargs = {'factr': 100}

def hinge(margin):
    """
    Classic hinge loss
    """
    return np.maximum(0, 1 - margin)


def hinge2(margin):
    """
    Looks pretty much like margin, but the corner is smoothed out
    """
    return np.log1p(np.exp(10 * (0.9 - margin))) / 10



def multi_svm_hinge_loss(x, y, weights, bias, alpha, pxw, pw_l2_sqr,
        l2_regularization, pl2_regularization):
    """
    x: (n_examples, n_features)
    y: (n_examples, n_classes)
    weights: (n_feaures, n_classes)
    bias: (n_classes,)
    alpha: (n_prev, n_classes)
    pxw: (n_examples, n_classes, n_prev)
    pw_l2_sqr: (n_prev, n_classes)
    l2_regularization: ()
    pl2_regularization: (n_prev,)
    """

    n_prev, n_classes = alpha.shape
    xw = dot(x, weights)
    if n_prev:
        assert pw_l2_sqr.shape == alpha.shape, (
                'pw_l2_sqr shape', pw_l2_sqr.shape,
                'alpha shape', alpha.shape)
        if np.any(pw_l2_sqr < 0):
            raise ValueError('prev_w_l2_sqr may not be negative')
        prev_l2_sqr = np.sum(
                pl2_regularization[:, None] * pw_l2_sqr * (alpha ** 2))
        n_examples, n_classes2, n_prev2 = pxw.shape
        assert n_prev2 == n_prev, ('n_prev', n_prev, n_prev2)
        assert n_classes2 == n_classes, ('n_classes', n_classes, n_classes2)
        prev_xw = (pxw * alpha.T).sum(axis=2)
        assert prev_xw.shape == xw.shape, ('xw', xw.shape, prev_xw.shape)
        xw = xw + prev_xw
    else:
        prev_l2_sqr = 0.0

    margin = y * (xw + bias )
    losses = hinge2(margin).mean(axis=0).sum()

    cur_l2_sqr = l2_regularization * (weights * weights).sum()
    l2_reg = 0.5 * (cur_l2_sqr + prev_l2_sqr)
    cost = losses + l2_reg
    return cost


def append_xw(pxw, x, weights):
    """
    Append dot(x, weights) to pxw
    """
    n_features, n_classes = weights.shape

    if x.size == 0:
        my_xw = np.zeros((len(x), n_classes, 1), dtype=pxw.dtype)
    elif np.all(weights == 0):
        my_xw = np.zeros((len(x), n_classes, 1), dtype=pxw.dtype)
    else:
        my_xw = dot(x, weights)[:, :, None]
    rval = np.concatenate([pxw, my_xw], axis=2)
    return rval.astype(pxw.dtype)


def append_alpha(alpha):
    n_prev, n_classes = alpha.shape
    ones = np.ones((1, n_classes), dtype=alpha.dtype)
    rval = np.vstack([alpha, ones])
    return rval.astype(alpha.dtype)


def append_w_l2_sqr(w_l2_sqr, weights):
    l2_sqr = (weights * weights).sum(axis=0)
    rval = np.vstack([w_l2_sqr, l2_sqr[None, :]])
    return rval.astype(w_l2_sqr.dtype)


def append_l2_regularization(pl2reg, l2reg):
    rval = np.hstack([pl2reg, [l2reg]])
    return rval.astype(pl2reg.dtype)


def fit_sgd_0(weights, bias, x, y, l2_regularization, n_iters,
        print_interval):
    """
    Refine `weights` and `bias` by n_iters steps of SGD
    """
    if n_iters <= 0:
        return weights, bias

    n_examples = len(x)
    n_features, n_classes = weights.shape
    alpha0 = np.empty((0, n_classes), dtype=weights.dtype)

    # -- use the first few elements of x to estimate the average
    #    example norm
    # -- fixing these learning rates makes sense to me because the
    #    hinge loss puts a bound on the slope of the function being
    #    optimized, the only variable is the norm / magnitude of the
    #    data.
    avg_w_norm = np.mean(np.sqrt((x[:200] ** 2).sum(axis=1)))
    step_size_w = 0.01 / (avg_w_norm + 1e-8)
    step_size_b = 0.01
    step_size_a = 0.0

    weights, bias, alpha0, = autodiff.fmin_sgd(
            lambda w, b, a, xx, yy1:
                multi_svm_hinge_loss(xx, yy1, w, b, a,
                    None, # xwi,
                    None, # prev_w_l2_sqr,
                    l2_regularization,
                    None),
            (weights, bias, alpha0),
            streams={
                'xx': x.reshape((n_examples, 1, n_features)),
                'yy1': y.reshape((n_examples, 1, n_classes)),
                },
            print_interval=print_interval,
            step_size=(step_size_w, step_size_b, step_size_a),
            step_size_backoff=0.1,
            loops=n_iters / float(len(x)),
            theano_mode=theano.Mode(
                linker='cvm_nogc',
                #linker='c|py',
                optimizer='fast_run').excluding('gpu'),
            theano_device='cpu',
            floatX=x.dtype,
            )
    return weights, bias


l_bfgs_b_debug_feature_limit = None

def fit_l_bfgs_b(weights, bias, alpha, x, y, l2reg,
        pxw, pw_l2_sqr, pl2reg, bfgs_kwargs,
        return_after_one_fit=False):
    """
    Refine `weights, bias, alpha` by l_bfgs_b
    """
    n_features, n_classes = weights.shape
    n_prev, n_classes = alpha.shape

    alpha_orig = alpha
    # -- the inplace alpha2 scaling modifies not-yet-fit weights
    #    as the while loop below works its way across
    weights = weights.copy()

    low = 0
    high = n_features

    # -- keep trying to train on less and less of the data until it works
    while True:
        x0 = x[:, low:high]

        x2 = x[:, high:]
        pxw2 = append_xw(pxw, x2, weights[high:])
        pl2reg2 = append_l2_regularization(pl2reg, l2reg)
        alpha2 = append_alpha(alpha)
        pw_l2_sqr2 = append_w_l2_sqr(pw_l2_sqr, weights[high:])

        def fn(w, b, a):
            return multi_svm_hinge_loss(x0, y, w, b, a,
                    pxw2, pw_l2_sqr2, l2reg, pl2reg2)
        try:
            if l_bfgs_b_debug_feature_limit is not None:
                # -- this mechanism is used by unit tests
                if (high - low) > l_bfgs_b_debug_feature_limit:
                    raise MemoryError()
            (weights_, bias, alpha2), info = autodiff.fmin_l_bfgs_b(fn,
                    (weights[low:high], bias, alpha2),
                    return_info=True,
                    borrowable=[x0],
                    floatX=x.dtype,
                    **bfgs_kwargs)
            info['feature_high'] = high
            info['feature_low'] = low
            gc.collect()
            logger.info('fitting successful for %i features' % high)
            break
        except (MemoryError, RuntimeError), e:
            high /= 2
            if low == high:
                raise
            gc.collect()
            logger.info('fitting required too much memory, falling back to %i' % high)
            continue

    weights[low:high] = weights_
    # -- pop off the alpha we just added
    weights[high:] *= alpha2[-1]
    alpha = alpha2[:-1].copy()

    if high == n_features or return_after_one_fit:
        return (weights, bias, alpha), [info]

    # -- now loop over all the features, and put the results together
    inc = high - low
    w0s = [weights_]
    costs = [info['fopt']]
    infos = [info]
    while high < n_features:
        high += inc
        low += inc

        x1 = x[:, low:high]
        pxw1 = append_xw(pxw, x0, weights_)
        pl2reg1 = append_l2_regularization(pl2reg, l2reg)
        alpha = append_alpha(alpha)
        pw_l2_sqr1 = append_w_l2_sqr(pw_l2_sqr, weights_)

        x2 = x[:, high:]
        pxw2 = append_xw(pxw1, x2, weights[high:])
        pl2reg2 = append_l2_regularization(pl2reg1, l2reg)
        alpha2 = append_alpha(alpha)
        pw_l2_sqr2 = append_w_l2_sqr(pw_l2_sqr1, weights[high:])

        def fn(w, b, a):
            return multi_svm_hinge_loss(x1, y, w, b, a,
                    pxw2, pw_l2_sqr2, l2reg, pl2reg2)
        (weights_, bias, alpha2), info = autodiff.fmin_l_bfgs_b(fn,
                (weights[low:high], bias, alpha2),
                return_info=True,
                borrowable=[x1],
                floatX=x.dtype,
                **bfgs_kwargs)

        info['feature_high'] = high
        info['feature_low'] = low

        # -- pop off the alpha we just added
        weights[high:] *= alpha2[-1]
        alpha = alpha2[:-1].copy()

        w0s.append(weights_)
        costs.append(info['fopt'])
        infos.append(info)
        x0 = x1
        pxw = pxw1
        pl2reg = pl2reg1
        pw_l2_sqr = pw_l2_sqr1

    old_alpha = alpha[:n_prev]
    new_alpha = alpha[n_prev:]
    assert len(new_alpha) == len(w0s) - 1

    if np.any(old_alpha < 0) or np.any(old_alpha > 1):
        warn('Alpha naturally grew beyond 0-1 range: %s' % str(old_alpha))

    for w, a in zip(w0s[:-1], new_alpha):
        w *= a
    weights = np.vstack(w0s)
    alpha_rval = old_alpha.copy()
    assert alpha_rval.shape == alpha_orig.shape
    return (weights, bias, alpha_rval), infos


class IncrementalMultiSVM(object):
    """
    On each iteration of the incremental construction this class fits a new
    weight vector w to the features x, while adjusting the norm of the
    previously-fit weight vectors to balance the current model against the old
    ones.

    See test_hingeboost.py for an example of incremental SVM construction.

    """

    def __init__(self, n_features, n_classes,
            prev_w_l2_sqr=None,
            l2_regularization=1e-4,
            prev_l2_regularization=None,
            dtype='float64',
            scalar_bounds=(-1e3, 1e3),
            bfgs_kwargs=None,
            alpha=None,
            print_interval=sys.maxint,
            n_sgd_iters=3000,
            bias=None,
            assert_clip_ok=True,
            badfit_thresh=float('inf'),
            ):

        self.n_features = n_features
        if prev_w_l2_sqr is None:
            self.prev_w_l2_sqr = np.empty((0, n_classes), dtype=dtype)
        else:
            self.prev_w_l2_sqr = np.asarray(prev_w_l2_sqr).astype(dtype)
        (self.n_prev, self.n_classes) = self.prev_w_l2_sqr.shape
        if n_classes != self.n_classes:
            raise ValueError('n_classes does not match prev_w_l2_sqr.shape',
                    n_classes, self.prev_w_l2_sqr.shape)
        self.l2_regularization = l2_regularization
        if prev_l2_regularization is None:
            self.prev_l2_regularization = np.empty((0,), dtype=dtype)
        else:
            self.prev_l2_regularization = prev_l2_regularization
        self.dtype = dtype
        self.scalar_bounds = scalar_bounds
        self.print_interval = print_interval
        if bfgs_kwargs is None:
            self.bfgs_kwargs = copy.deepcopy(_default_bfgs_kwargs)
            if print_interval < sys.maxint:
                self.bfgs_kwargs.setdefault('iprint', 1)
        else:
            self.bfgs_kwargs = bfgs_kwargs

        self.weights = np.zeros((n_features, n_classes), dtype=dtype)
        if bias is None:
            self.bias = np.zeros((n_classes,), dtype=dtype)
        else:
            self.bias = np.asarray(bias).astype(dtype)
            if (n_classes,) != self.bias.shape:
                raise ValueError('bad shape for bias', self.bias.shape)
        if alpha is None:
            self.alpha = np.ones_like(self.prev_w_l2_sqr)
        else:
            self.alpha = np.array(alpha).astype(dtype)
        if self.alpha.shape != self.prev_w_l2_sqr.shape:
            raise ValueError('shape mismatch between alpha and prev_w_l2_sqr',
                    self.alpha.shape, self.prev_w_l2_sqr.shape)
        self.n_sgd_iters = n_sgd_iters
        self.assert_clip_ok = assert_clip_ok
        self.badfit_thresh = badfit_thresh

    def print_summary(self):
        print 'IncrementalMultiSVM', repr(self)
        print '-> alpha', self.alpha
        print '-> prvl2', self.prev_l2_regularization
        print '-> prvw2', self.prev_w_l2_sqr

    @property
    def cumulative_alpha(self):
        return append_alpha(self.alpha)

    @property
    def cumulative_w_l2_sqr(self):
        return append_w_l2_sqr(self.prev_w_l2_sqr, self.weights)

    @property
    def cumulative_l2_regularization(self):
        return append_l2_regularization(self.prev_l2_regularization,
                self.l2_regularization)

    def xw_carry_forward(self, x, pxw=None):
        return append_xw(self.as_xw(x, pxw), x, self.weights)

    def continuation(self, n_features=None, l2_regularization=None):
        if n_features is None:
            n_features = self.n_features
        if l2_regularization is None:
            l2_regularization = self.l2_regularization

        rval = self.__class__(
                n_features=n_features,
                n_classes=self.n_classes,
                prev_w_l2_sqr=self.cumulative_w_l2_sqr,
                alpha=self.cumulative_alpha,
                prev_l2_regularization=self.cumulative_l2_regularization,
                l2_regularization=l2_regularization,
                dtype=self.dtype,
                scalar_bounds=self.scalar_bounds,
                print_interval=self.print_interval,
                bfgs_kwargs=self.bfgs_kwargs,
                n_sgd_iters=self.n_sgd_iters,
                bias=self.bias.copy(),
                assert_clip_ok=self.assert_clip_ok,
                )
        return rval

    def decision_function(self, x, xw=None):
        rval = dot(x, self.weights) + self.bias
        xw = self.as_xw(x, xw)
        if xw.size or self.alpha.size:
            # -- workaround Theano's no support for tensordot
            rval += (xw * self.alpha.T).sum(axis=2)
        return rval

    def as_xw(self, x, xw):
        if xw is None:
            if self.n_prev == 0:
                return np.zeros(
                        (len(x), self.n_classes, self.n_prev),
                        dtype=x.dtype)
            else:
                raise TypeError('xw is required for previous models')
        else:
            xw = np.asarray(xw, dtype=self.dtype, order='C')
            if xw.shape != (len(x), self.n_classes, self.n_prev):
                raise ValueError('xw has wrong shape',
                        (xw.shape, (len(x), self.n_classes, self.n_prev)))
            return xw

    def predict(self, x, xw=None):
        xw = self.as_xw(x, xw)
        return self.decision_function(x, xw).argmax(axis=1)

    def y_ind(self, y):
        # y_ind is all +-1, with 1 meaning a positive label for OvA classif
        assert y.min() == 0 # fail for +-1 labels
        y_ind = -np.ones((len(y), self.n_classes)).astype(self.dtype)
        y_ind[np.arange(len(y)), y] = 1
        return y_ind

    def loss(self, x, y, xw=None):
        xw = self.as_xw(x, xw)
        y_ind = self.y_ind(y)
        assert self.l2_regularization is not None
        return multi_svm_hinge_loss(x, y_ind,
                self.weights, self.bias, self.alpha,
                xw,
                self.prev_w_l2_sqr,
                self.l2_regularization,
                self.prev_l2_regularization,
                )

    def fit(self, x, y, xw=None):
        """
        x - n_examples x n_features design matrix.
        y - vector of integer labels
        xw - matrix of real-valued incoming biases obtained
            by multiplying the existing weight vectors by x
        """
        pxw = self.as_xw(x, xw)
        assert y.min() == 0 # fail for +-1 labels

        if x.shape[0] != y.shape[0]:
            raise ValueError('length mismatch between x and y')

        n_examples, n_classes, n_prev = pxw.shape
        if n_prev != self.n_prev:
            raise ValueError('n_prev mismatch',
                    (n_prev, self.n_prev))
        if n_examples != len(x):
            raise ValueError('n_examples mismatch',
                    (n_examples, len(x)))
        if n_classes != self.weights.shape[1]:
            raise ValueError('n_classes mismatch',
                    (n_classes, self.weights.shape[1]))

        weights = self.weights
        bias = self.bias
        alpha = self.alpha

        bias0 = np.zeros_like(bias)
        alpha0 = np.empty((0, self.n_classes), dtype=alpha.dtype)

        y_ind = self.y_ind(y)

        bfgs_kwargs = dict(self.bfgs_kwargs)
        bfgs_kwargs.setdefault('factr', 100)

        bfgs_kwargs_precise = dict(bfgs_kwargs)
        bfgs_kwargs_precise['factr'] /= 100

        # -- warm up with some pure-online sgd
        #    don't train alpha yet, wait until the weights and bias
        #    are somewhat initialized.
        weights, bias = fit_sgd_0(weights, bias, x, y_ind,
                self.l2_regularization,
                self.n_sgd_iters,
                self.print_interval)

        c0 = n_prev / (1.0 + n_prev)
        c1 = 1 / (1.0 + n_prev)

        alpha *= c0
        weights *= c1
        bias = c0 * bias0 + c1 * bias

        (p_weights, p_bias, p_alpha), infos = fit_l_bfgs_b(
                weights, bias, alpha,
                x, y_ind, self.l2_regularization,
                pxw,
                self.prev_w_l2_sqr,
                self.prev_l2_regularization,
                self.bfgs_kwargs,
                return_after_one_fit=True)

        if infos[0]['feature_high'] == self.n_features:
            # -- the first fit did the whole feature set
            weights = p_weights
            bias = p_bias
            alpha = p_alpha
        elif infos[0]['fopt'] >= self.badfit_thresh:
            # -- the first fit was so bad that we're giving up
            weights = p_weights
            bias = p_bias
            alpha = p_alpha
        else:
            # -- we couldn't fit the whole feature set at once
            data_home = skdata.data_home.get_data_home()
            tempdirname = os.path.join(data_home, 'hpconvnet_isvm_features')
            if not os.path.exists(tempdirname):
                os.makedirs(tempdirname)
            dirname = tempfile.mkdtemp(dir=tempdirname)
            try:
                README = open(os.path.join(dirname, 'README'), 'w+')
                print >> README, (
                    "Feature cache created by hpconvnet/isvm_multi.py")
                README.close()
                p_x = np.memmap(os.path.join(dirname, 'p_x.npy'),
                        dtype=x.dtype,
                        mode='w+',
                        shape=x.shape)

                for ii in range(2):
                    # -- if there isn't enough GPU memory to fit the whole
                    #    problem at once, then use a block coordinate descent
                    #    strategy, with different blocks on each iteration.
                    #    I found that 2 passes of this kind were sufficient
                    #    for MNIST, when divided into 2 pieces.

                    perm = np.random.RandomState(1234 + ii).permutation(
                            self.n_features)

                    p_weights = weights[perm]
                    for ii in xrange(len(x)):
                        x_ii = x[ii] * 1  # -- bring it into memory
                        p_x[ii] = x_ii[perm]

                    (p_weights, bias, alpha), infos2 = fit_l_bfgs_b(
                            p_weights, bias, alpha,
                            p_x, y_ind, self.l2_regularization,
                            pxw,
                            self.prev_w_l2_sqr,
                            self.prev_l2_regularization,
                            self.bfgs_kwargs)

                    weights[perm] = p_weights
                    infos.extend(infos2)

            finally:
                shutil.rmtree(dirname)

        self.weights = weights
        self.bias = bias
        self.alpha = alpha
        self.fit_infos = infos

        # -- in cases where the prev_l2_sqr or the prev_l2_regularization are
        # really tiny, alpha can do funny things, like grow greater than 1,
        # and/or even be slightly negative.
        clipped_alpha = np.clip(alpha, 0, 1)
        if self.assert_clip_ok:
            final_loss = self.loss(x, y, xw)
            self.alpha = clipped_alpha
            clipped_final_loss = self.loss(x, y, xw)

            if not np.allclose(final_loss, clipped_final_loss, atol=1e-3,
                    rtol=1e-2):
                error('fit is significantly degraded by alpha-clipping')
                error('-> orig loss %f' % final_loss)
                error('-> clipped loss %f' % clipped_final_loss)
                error('-> alpha %s' % str(alpha))
        else:
            self.alpha = clipped_alpha



