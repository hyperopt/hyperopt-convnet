"""
This file supports the incremental construction of an SVM classifier
by partially-corrective boosting on the hinge loss.


Each incremental solver minimizes

    hinge( dot(X, w) + b + alpha * prev_Wx)
    + lambda * (|w|^2 + |alpha * prev_W|^2)


Each solver is designed to be run on a subset of all available features.

"""
import copy

import numpy as np
import autodiff
import theano # abstraction leak to pass mode for optimization

from isvm_multi import hinge2
from isvm_multi import _default_bfgs_kwargs
from isvm_multi import IncrementalMultiSVM

def binary_svm_hinge_loss(x, y, weights, bias, alpha, pxw, pw_l2_sqr,
        l2_regularization):
    """
    x: (n_examples, n_features) feature matrix
    y: (n_examples,) label vector
    weights: (n_features,) new weights
    alpha: (n_prev,) multiplier on each vector of previous weights
    pw_l2_sqr: (n_prev,) squared l2-norm of existing weights
    pxw: (n_examples, n_prev) inner products of previous weights with `x`
    l2_regularization: multiplier on sum of all squared weights
    """
    n_prev, = alpha.shape

    xw = np.dot(x, weights)
    if n_prev:
        if np.any(pw_l2_sqr < 0):
            raise ValueError('prev_w_l2_sqr may not be negative')
        prev_l2_sqr = (pw_l2_sqr * (alpha ** 2)).sum()
        xw += np.dot(pxw, alpha)
    else:
        prev_l2_sqr = 0.0

    margin = y * (xw + bias)
    losses = hinge2(margin)

    cur_l2_sqr = (weights * weights).sum()
    l2_reg = 0.5 * l2_regularization * (cur_l2_sqr + prev_l2_sqr)
    cost = losses.sum() + l2_reg
    return cost


class IncrementalSVM(object):
    """
    On each iteration of the incremental construction this class fits a new
    weight vector w to the features x, while adjusting the norm of the
    previously-fit weight vectors to balance the current model against the old
    ones.

    See test_hingeboost.py for an example of incremental SVM construction.

    """
    def __init__(self, n_features,
            prev_w_l2_sqr=None,
            l2_regularization=1e-4,
            dtype='float64',
            scalar_bounds=(-1e3, 1e3),
            bfgs_kwargs=None,
            alpha=None,
            ):
        """
        prev_w_l2_sqr: the (un-squared) l2-norm of each column of the existing weight vector
        """
        self.n_features = n_features
        if prev_w_l2_sqr is None:
            self.prev_w_l2_sqr = np.empty((0,), dtype=dtype)
        else:
            self.prev_w_l2_sqr = np.asarray(prev_w_l2_sqr).astype(dtype)
        (self.n_prev,) = self.prev_w_l2_sqr.shape
        self.l2_regularization = l2_regularization
        self.dtype = dtype
        self.scalar_bounds = scalar_bounds
        if bfgs_kwargs is None:
            self.bfgs_kwargs = copy.deepcopy(_default_bfgs_kwargs)
        else:
            self.bfgs_kwargs = bfgs_kwargs

        self.weights = np.zeros((n_features,), dtype=dtype)
        self.bias = np.zeros((), dtype=dtype)
        if alpha is None:
            self.alpha = np.ones_like(self.prev_w_l2_sqr)
        else:
            self.alpha = np.array(alpha).astype(dtype)
        if self.alpha.shape != self.prev_w_l2_sqr.shape:
            raise ValueError('shape mismatch between alpha and prev_w_l2_sqr',
                    self.alpha.shape, self.prev_w_l2_sqr.shape)

    @property
    def cumulative_alpha(self):
        rval = list(self.alpha)
        rval.append(1.0)
        return np.asarray(rval, dtype=self.dtype)

    @property
    def cumulative_w_l2_sqr(self):
        rval = list(self.prev_w_l2_sqr)
        rval.append(self.w_l2_sqr)
        return np.asarray(rval, dtype=self.dtype)

    @property
    def w_l2_sqr(self):
        return (self.weights * self.weights).sum()

    def continuation(self, n_features=None):
        if n_features is None:
            n_features = self.n_features
        rval = self.__class__(
                n_features=n_features,
                prev_w_l2_sqr=self.cumulative_w_l2_sqr,
                alpha=self.cumulative_alpha,
                l2_regularization=self.l2_regularization,
                dtype=self.dtype,
                scalar_bounds=self.scalar_bounds,
                bfgs_kwargs=self.bfgs_kwargs
                )
        rval.bias = self.bias.copy()
        return rval

    def xw_carry_forward(self, x, pxw=None):
        """stack the current dot(x, weights) onto previous stack `pxw`
        """
        pxw = self.as_xw(x, pxw)
        rval = np.hstack((pxw, np.dot(x, self.weights)[:, None]))
        print rval.shape
        return rval

    def decision_function(self, x, xw=None):
        xw = self.as_xw(x, xw)
        return np.dot(x, self.weights) + np.dot(xw, self.alpha) + self.bias

    def predict(self, x, xw=None):
        xw = self.as_xw(x, xw)
        return (self.decision_function(x, xw) > 0) * 2 - 1

    def loss(self, x, y, xw=None):
        # y_ind is all +-1, with 1 meaning a positive label for OvA classif
        assert set(y) <= set([-1, 1])
        xw = self.as_xw(x, xw)

        return binary_svm_hinge_loss(x, y,
                self.weights, self.bias, self.alpha,
                xw,
                self.prev_w_l2_sqr,
                self.l2_regularization,
                )

    def as_xw(self, x, xw):
        if xw is None:
            if self.n_prev == 0:
                return np.zeros((len(x), self.n_prev), dtype=x.dtype)
            else:
                raise TypeError('xw is required for previous models')
        else:
            xw = np.asarray(xw, dtype=self.dtype)
            if xw.shape != (len(x), self.n_prev):
                raise ValueError('xw has wrong shape',
                        xw.shape, (len(x), self.n_prev))
            return xw

    def fit(self, x, y, xw=None):
        """
        x - n_examples x n_features design matrix.
        y - vector of integer labels
        xw - matrix of real-valued incoming biases obtained
            by multiplying the existing weight vectors by x
        """
        assert set(y) <= set([-1, 1])

        if x.shape[0] != y.shape[0]:
            raise ValueError('length mismatch between x and y')
        n_examples, n_features = x.shape
        if n_features != self.n_features:
            raise ValueError('n_feature mismatch', (n_features,
                self.n_features))

        weights = self.weights
        bias = self.bias
        alpha = self.alpha

        x = x.astype(self.dtype)
        y = y.astype(self.dtype)

        xw = self.as_xw(x, xw)
        print 'WARNING: IncrementalSVM should use alpha0, n_sgd_iters'

        # -- warm up with some sgd
        weights, bias, alpha, = autodiff.fmin_sgd(
                lambda w, b, a, xi, yi, xwi:
                    binary_svm_hinge_loss(xi, yi, w, b, a, None,
                        None,
                        self.l2_regularization),
                (weights, bias, alpha),
                streams={
                    'xi': x.reshape((n_examples, 1, x.shape[1])),
                    'yi': y.reshape((n_examples, 1)),
                    },
                stepsize=0.01,
                loops=max(1, 100000 // len(x)),
                )

        # -- fine-tune without alpha by L-BFGS
        weights, bias, alpha, = autodiff.fmin_l_bfgs_b(
                lambda w, b, a:
                    binary_svm_hinge_loss(x, y,
                        w, b, a, None, None,
                        self.l2_regularization),
                (weights, bias, alpha),
                # -- the graph is tiny, time spent optimizing it is wasted.
                theano_mode=theano.Mode(linker='cvm', optimizer='fast_run'),
                **self.bfgs_kwargs)


        self.weights = weights
        self.bias = bias
        self.alpha = alpha


class IncrementalSVM_MultiHack(object):

    def __init__(self, l2_regularization):
        self.l2_regularization = l2_regularization

    def fit(self, x, y, history):
        self._svm = IncrementalMultiSVM(
                dtype=x.dtype,
                n_features=x.shape[1],
                n_classes=2,
                l2_regularization=self.l2_regularization,
                n_sgd_iters=0,
                bfgs_kwargs={
                    'maxfun': 1000,
                    'iprint': 0,
                    'm': 32,
                    'factr': 100},
                )
        self._svm.fit(x, (y + 1) / 2, history)

    def predict(self, x, history):
        return self._svm.predict(x, history) * 2 - 1
