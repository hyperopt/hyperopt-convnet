import copy
import logging
import functools

import numpy as np
import hyperopt
from sklearn.svm import SVC
from .utils import linear_kernel

logger = logging.getLogger(__name__)


class EnsembleSVC(object):
    """Fit and back-fit SVM ensemble.

    Without a `history` this solves cost function

    :math:`1/N \sum_i hinge(y_i max(1 - w \cdot x_i + b)) + \alpha ||w||^2`

    Parameters
    ----------
    """

    def __init__(self, train_sample='train'):
        self.train_sample = train_sample

        self._grams = {}
        self._weights = {}
        self._svm = None
        self._labels = {}
        self._features = {}
        self._compound_samples = {}
        self._kernels = {}

    def copy(self):
        rval = self.__class__(train_sample=self.train_sample)
        rval._grams = dict(self._grams)
        rval._weights = copy.deepcopy(self._weights)
        rval._svm = copy.deepcopy(self._svm)
        rval._labels = copy.deepcopy(self._labels)
        rval._features = dict(self._features)
        rval._compound_samples = copy.deepcopy(self._compound_samples)
        rval._kernels = copy.deepcopy(self._kernels)
        return rval

    @property
    def members(self):
        return self._weights.keys()

    def has_member(self, member):
        return member in self._weights

    def add_member(self, member, weight=1.0, kernel=linear_kernel):
        logger.debug('add_member: %s' % member)
        if member in self._weights:
            if (self._weights[member] == weight
                    and self._kernels[member] == kernel):
                pass
            else:
                raise KeyError(member)
        else:
            self._weights[member] = weight
            self._kernels[member] = kernel

    def set_weight(self, member, weight):
        self._weights[member] = weight

    def add_sample(self, sample, labels=None):
        logger.debug('add_sample: %s' % sample)
        if sample in self._labels:
            if np.all(labels == self._labels[sample]):
                pass
            else:
                raise KeyError(sample)
        else:
            self._labels[sample] = labels

    def add_compound_sample(self, sample, subsamples):
        logger.debug('add_compound_sample: %s' % sample)
        if sample in self._compound_samples:
            raise KeyError(sample)
        else:
            if not isinstance(subsamples, (list, tuple)):
                raise TypeError(subsamples)
            self._compound_samples[sample] = subsamples

    def as_raw_samples(self, sample1):
        if isinstance(sample1, (tuple, list)):
            def add(a, b):
                return a + b
            return reduce(add, map(self.as_raw_samples, sample1))
        elif sample1 in self._compound_samples:
            return self.as_raw_samples(self._compound_samples[sample1])
        else:
            return [sample1]

    def add_features(self, member, sample, features):
        if member not in self._weights:
            raise KeyError(member)
        if sample not in self._labels:
            raise KeyError(sample)
        self._features[(member, sample)] = features

    def del_features(self, member, sample):
        del self._features[(member, sample)]

    def has_gram(self, member, sample1, sample2):
        return (member, sample1, sample2) in self._grams

    def add_gram(self, member, sample1, sample2, gram):
        if member not in self._weights:
            raise KeyError(member)
        if sample1 not in self._labels:
            raise KeyError(sample1)
        if sample2 not in self._labels:
            raise KeyError(sample2)
        logger.debug('add_gram: (%s, %s, %s) -> (%i, %i) array of %s' % (
            member, sample1, sample2,
            gram.shape[0], gram.shape[1], gram.dtype))
        self._grams[(member, sample1, sample2)] = gram
        self._grams[(member, sample2, sample1)] = gram.T

    def del_gram(self, member, sample1, sample2):
        del self._grams[(member, sample1, sample2)]
        del self._grams[(member, sample2, sample1)]

    def compute_gram(self, member, sample1, sample2, dtype=np.float):
        if member not in self._weights:
            raise KeyError(member)
        if sample1 not in self._labels:
            raise KeyError(sample1)
        if sample2 not in self._labels:
            raise KeyError(sample2)
        f1 = self._features[(member, sample1)]
        f2 = self._features[(member, sample2)]
        gram = self._kernels[member](f1, f2).astype(dtype)
        if gram.shape != (len(f1), len(f2)):
            raise ValueError('kernel function returned wrong shape')
        self.add_gram(member, sample1, sample2, gram)

    def compute_all_grams(self, members, samples):
        for member in members:
            for sample1 in samples:
                for sample2 in samples:
                    if (member, sample1, sample2) not in self._grams:
                        self.compute_gram(member, sample1, sample2)

    def gram(self, member, sample1, sample2):
        to_vstack = []
        row_samples = self.as_raw_samples(sample1)
        col_samples = self.as_raw_samples(sample2)
        for rs in row_samples:
            to_hstack = [self._grams[(member, rs, cs)]
                    for cs in col_samples]
            to_vstack.append(np.hstack(to_hstack))
        rval = np.vstack(to_vstack)
        return rval

    def labels(self, sample):
        raw_samples = self.as_raw_samples(sample)
        raw_labels = [self._labels[s] for s in raw_samples]
        return np.concatenate(raw_labels)

    def weighted_gram(self, sample1, sample2, weights=None):
        if weights is None:
            weights = self._weights
        members = weights.keys()
        # -- sorting not strictly necessary, but helps different processes to
        # -- perform the same calculation.
        members.sort()
        rval = None
        for m in members:
            # -- The weights represent squared importance coefficients, one
            # -- on each ensemble member.
            gg = weights[m] * self.gram(m, sample1, sample2)
            rval = gg if rval is None else gg + rval
        return rval

    def fit_svm(self, l2_regularization=None, train_sample=None):
        if train_sample is None:
            train_sample = self.train_sample

        g_trn = self.weighted_gram(train_sample, train_sample)

        if l2_regularization is None:
            l2_regularization = 1.0 / len(g_trn)

        C = 1.0 / (l2_regularization * len(g_trn))

        if l2_regularization is None:
            assert np.allclose(C, 1.0)
        svm = SVC(
            C=C,
            kernel='precomputed',
            cache_size=1.1 * 4.0 * g_trn.size / (1 << 20),
            max_iter=0.5 * len(g_trn) ** 2,  # COMPLETE HEURISTIC GUESS
            )
        svm.fit(g_trn, self.labels(train_sample))
        self._svm = svm

    def fit_weights_crossvalid(self, validation_sample, max_evals,
        algo=None,
        scales=100.0,
        members=None,
        ):
        """Fit an SVM and optimize [some of] the kernel weights.

        Parameters
        ----------
        validation_sample : sample identifier
            Adjust hyperparameters to optimize performance on this set.

        max_evals : integer
            Try no more than this many hyperparameter settings.

        algo: hyperopt.algo
            A hyperopt optimization algorithm for hyperparameters.
            Default is currently hyperopt.tpe.suggest

        scales: float or dict: member -> float
            Multiplicative uncertainty around the current weight value
            for each member (larger for broader search).

        members : None or sequence of member names
            Members to combine as in MKL. `None` means to use all members.


        TODO
        ----
        Look at literature for MKL and do something more efficient
        and accurate.
        """

        # -- N.B.
        # -- We don't need to take l2-regularization into account because by
        # -- optimizing the norm of the weights, we are implicitly optimizing
        # -- the l2-regularization on the model.

        trn_sample = self.train_sample
        val_sample = validation_sample

        labels_trn = self.labels(trn_sample)
        labels_val = self.labels(val_sample)

        if algo is None:
            algo = functools.partial(
                        hyperopt.tpe.suggest,
                        n_startup_jobs=5)

        if isinstance(scales, (int, float, np.number)):
            #scales = {m: scales for m in self._weights}
            scales = dict([(m, scales) for m in self._weights])
        else:
            if set(scales.keys()) != set(self._weights.keys()):
                raise ValueError('wrong number of search scales')

        if members is None:
            members = self._weights.keys()
        else:
            members = list(members)
        # -- sorting not strictly necessary, but helps different processes to
        # -- perform the same calculation by presenting the same `domain` below.
        members.sort()

        def eval_weights(ws):
            ws = np.asarray(ws)
            ws_weights = copy.deepcopy(self._weights)
            ws_weights.update(dict(zip(members, ws)))
            g_trn = self.weighted_gram(trn_sample, trn_sample, ws_weights)
            g_val = self.weighted_gram(val_sample, trn_sample, ws_weights)

            logger.debug('fitting ws=%s' % str(ws))
            svm = SVC(
                C=1.0,
                kernel='precomputed',
                cache_size=1.1 * 4.0 * g_trn.size / (1 << 20),
                max_iter=0.5 * len(g_trn) ** 2,  # XXX: COMPLETE HEURISTIC GUESS
                )

            TINY = 1e-8
            def nogood(msg):
                logger.debug('f(%s) -> "%s"' % (ws, msg))
                return dict(loss=1.0, status='ok', svm=svm, fit=False)

            if not np.all(np.isfinite(g_trn)):
                return nogood('non-finite gram matrix (train)')
            if np.all(abs(g_trn) < TINY):
                return nogood('null gram matrix (train)')
            if not np.all(np.isfinite(g_val)):
                return nogood('non-finite gram matrix (valid)')
            if np.all(abs(g_val) < TINY):
                return nogood('null gram matrix (valid)')

            svm.fit(g_trn, labels_trn)
            logger.debug('done!')
            pred_val = svm.predict(g_val)
            assert labels_val.shape == pred_val.shape
            err_rate = np.mean(labels_val != pred_val)
            # XXX: to break ties, take smaller weights
            rval = err_rate + 1e-4 * np.log1p(np.sum(ws))
            logger.debug('f(%s) -> %f -> %f' % (ws, err_rate, rval))
            return dict(loss=rval, status='ok', svm=svm, fit=True)

        try:
            # -- This optimizer seems a little bit less finicky than the
            # scipy ones (!?) such as anneal (doesn't respect lower bound,
            # or maxeval) and slsqp (xvalid cost is not continuous)... but
            # -- TODO: when a GP-based optimizer is in hyperopt, that would
            # probably work even better.
            # -- Note: I emailed Jasper about pushing his GP impl to sklearn
            level = logging.getLogger('hyperopt').level
            logging.getLogger('hyperopt').setLevel(logging.WARN)
            trials = hyperopt.Trials()
            domain = [
                hyperopt.hp.lognormal(
                    str(m), np.log(self._weights[m]), np.log(scales[m]))
                for m in members]
            hyperopt.fmin(
                eval_weights,
                space=domain,
                trials=trials,
                max_evals=max_evals,
                algo=algo,
                )
        finally:
            logging.getLogger('hyperopt').setLevel(level)
        final_weights = trials.argmin

        if not trials.best_trial['result']['fit']:
            # -- meant to be caught by slm.py call_catching_pipeline_errors()
            raise ValueError('Failed to fit SVM (non-finite features)')
        self._svm = trials.best_trial['result']['svm']
        self._weights = final_weights

    def predict(self, test_sample):
        g = self.weighted_gram(test_sample, self.train_sample)
        return self._svm.predict(g)

    def error_rate(self, test_sample):
        preds = self.predict(test_sample)
        rval = np.mean(preds != self._labels[test_sample])
        return rval

    def print_summary(self):
        print 'EnsembleSVC.print_summary()'
        print '  weights', self._weights

