import copy
import cPickle
import functools
import logging
import os

import numpy as np

from hyperopt import pyll
import hyperopt
from hyperopt.base import use_obj_for_literal_in_memo

from skdata.larray import cache_memmap

#from .isvm_boosting import BoostHelper
from .isvm_multi import IncrementalMultiSVM
from .pyll_slm import error_rate
from .pyll_slm import view2_worth_calculating
from .slm_visitor import SLM_Visitor
from .slm import USLM_Exception
from .slm import call_catching_pipeline_errors
from .utils import git_versions

logger = logging.getLogger(__name__)
info = logger.info
warn = logger.warn

loads = cPickle.loads
dumps = functools.partial(cPickle.dumps, protocol=-1)


def recupdate(dct, path, payload):
    if path:
        if not isinstance(path[0], basestring):
            raise TypeError(path[0])
        dct.setdefault(path[0], {})
        return recupdate(dct[path[0]], path[1:], payload)
    else:
        dct.update(payload)
        return dct


class PrimalVisitor(SLM_Visitor):
    """
    An skdata-compatible learning algorithm that implements SVM with
    isvm_binary and isvm_multi.

    This class takes an *evaluated* pipeline as a starting point,
    and applies it to a data set.  Nothing symbolic here.
    """

    def __init__(self,
        optimize_l2_reg,
        thresh_loss=None,
        thresh_rank=10,
        badfit_thresh=float('inf'),
        svm_crossvalid_max_evals=20,
        **kwargs
        ):
        SLM_Visitor.__init__(self, **kwargs)

        self.optimize_l2_reg = optimize_l2_reg
        self.member_name = self._member_name()
        self.thresh_loss = thresh_loss
        self.thresh_rank = thresh_rank
        self.svm_crossvalid_max_evals = svm_crossvalid_max_evals
        self.badfit_thresh = badfit_thresh

        self._results = {}
        self._obj_results = {}

        if not self.optimize_l2_reg:
            raise NotImplementedError()

    def hyperopt_rval(self):
        return copy.deepcopy(self._results)

    def attach_obj_results(self):
        ctrl = self.ctrl
        att = ctrl.trials.trial_attachments(ctrl.current_trial)
        def helper(dct, prefix):
            for key, val in dct.items():
                if isinstance(val, dict):
                    helper(val, '%s/%s' % (prefix, key))
                else:
                    att['%s/%s' % (prefix, key)] = dumps(val)
        helper(self._obj_results, '')

    def trial_obj_attachment(self, trial, rpath):
        key = '/' + '/'.join(rpath)
        att = self.ctrl.trials.trial_attachments(trial)
        msg = att[key]
        rval = loads(msg)
        return rval

    def add_results(self, path, simple, objs):
        for key, val in objs.items():
            if isinstance(val, dict):
                raise TypeError('cannot attach a dict', key)
        recupdate(self._results, path, simple)
        recupdate(self._obj_results, path, objs)

    def load_history(self):
        raise NotImplementedError()

    def load_svm(self, train_name, valid_name,
                      n_features, n_classes, l2_reg):
        if self.history:
            prev_doc = self.history[-1]
            info('load_svm: %i previous model documents found'
                    % len(self.history))
            info('load_svm: Most-previous model document tid: %s'
                    % prev_doc['tid'])
            att = self.ctrl.trials.trial_attachments(prev_doc)
            prev_svm = self.trial_obj_attachment(prev_doc,
                ['train_indexed_image_classification', train_name,
                    valid_name, 'model'])
            svm = prev_svm.continuation(n_features, l2_reg)
            info('load_svm: alpha shape %s'
                    % str(svm.alpha.shape))
            info('load_svm: prev_w_l2_sqr shape %s'
                    % str(svm.prev_w_l2_sqr.shape))
        else:
            info('load_svm: No previous model document found')
            info('load_svm: Allocating SVM for %i x %i problem'
                    % (n_features, n_classes))
            svm = IncrementalMultiSVM(n_features, n_classes,
                    l2_regularization=l2_reg,
                    dtype='float32',
                    # --  TODO consider maxfun, M, tolerances?
                    bfgs_kwargs={
                        'maxfun': 1000,
                        'iprint': 0,
                        'm': 32,
                        'factr': 100},
                    print_interval=5000,
                    n_sgd_iters=0,
                    badfit_thresh=self.badfit_thresh,
                    )
        return svm

    def load_prev_xw(self, task_name, train_name, valid_name, use_history):
        assert use_history in ('using_history', 'not_using_history')
        if not self.history:
            info('load_prev_xw: No previous model documents for %s/%s/%s'
                    % (task_name, train_name, valid_name))
            return None
        prev_xw_list = []
        for pm_doc in self.history:
            info('load_prev_xw doc %i loss %f' % (
                pm_doc['tid'], pm_doc['result']['loss']))
            xw = self.trial_obj_attachment(pm_doc,
                    ['loss_indexed_image_classification',
                        task_name, train_name, valid_name, use_history, 'xw'])
            prev_xw_list.append(xw.astype('float32'))

        info('load_prev_xw: %i previous model documents found'
                % len(prev_xw_list))
        # -- put them into desired shape: (examples, classes, models)
        prev_xw = np.asarray(prev_xw_list).transpose(1, 2, 0).copy()
        return prev_xw

    def train_indexed_image_classification(self, train, valid=None):

        if valid is None:
            train_name = train.name
            valid_name = 'None'
        else:
            train_name = train.name
            valid_name = valid.name
            assert train.all_images is valid.all_images
            assert train.all_labels is valid.all_labels

        info('train_indexed_image_classification: %s/%s' % (
            train_name, valid_name))

        normed_features, xmean, xstd, avg_nrm = \
            self.normalized_image_features(
                train.all_images, None, None, None, flatten=True)

        assert train.name is not None

        if hasattr(self, 'cmemmap'):
            assert train.all_images is self.cmemmap_all_images
        else:
            self.cmemmap_all_images = train.all_images
            self.cmemmap = cache_memmap(
                normed_features,
                self.memmap_name,
                del_atexit=True)

        if not hasattr(self, 'history'):
            self.load_ensemble_history(fields=[])

        svm = self.load_svm(
            train_name, valid_name, self.cmemmap.shape[1],
            train.n_classes, self.pipeline['l2_reg'])
        svm.feature_xmean = xmean
        svm.feature_xstd = xstd
        svm.feature_avg_nrm = avg_nrm
        svm.train_name = train_name
        svm.valid_name = valid_name

        prev_xw_trn = self.load_prev_xw(
            train_name, train_name, valid_name, use_history='using_history')

        info('train_indexed_image_classification: Fitting SVM with prev_xw')
        svm.fit(self.cmemmap[train.idxs],
                train.all_labels[train.idxs],
                prev_xw_trn)

        info('-> loaded alpha %s' % str(svm.alpha))
        info('-> loaded prvl2 %s' % str(svm.prev_l2_regularization))
        info('-> loaded prvw2 %s' % str(svm.prev_w_l2_sqr))

        if valid is None:
            # -- XXX: it is currently a hack to use the existence
            #    of the validation set to decide when to compute
            #    an svm without the history features... it currently
            #    so happens that for the fit/val split we have a validation
            #    set and we want to train both ways, and for the sel/test
            #    split we do not have a validation set and we only want the
            #    fit-with-history training.
            assert train.name == 'sel'
            svm0 = None
        else:
            svm0 = copy.deepcopy(svm)
            if (prev_xw_trn is not None) and prev_xw_trn.size:
                info('Fitting SVM without prev_xw')
                svm0.fit(self.cmemmap[train.idxs],
                         train.all_labels[train.idxs],
                         np.zeros_like(prev_xw_trn))
        self.add_results(
            [
            'train_indexed_image_classification',
            train_name,
            valid_name,
            ],
            {
            'train_name': train_name,
            'valid used': (valid is not None),
            'valid_name': valid_name,
            },
            {
            'model0': svm0,
            'model': svm,
            })

        self.loss_indexed_image_classification(svm, train)
        if valid is not None:
            self.loss_indexed_image_classification(svm, valid)
            self.loss_indexed_image_classification(svm0, valid,
                use_history='not_using_history')

        return svm

    def loss_indexed_image_classification(self, model, task,
        use_history='using_history'):
        assert task.name

        # -- N.B. using_history here, because we want to build on the models
        # that *were* using history
        prev_xw = self.load_prev_xw(task.name,
                model.train_name, model.valid_name,
                use_history='using_history')

        x = self.cmemmap[task.idxs]

        if (use_history == 'using_history') or (prev_xw is None):
            preds = model.predict(x, prev_xw)
        else:
            preds = model.predict(x, np.zeros_like(prev_xw))
        erate = error_rate(preds, task.all_labels[task.idxs])
        xw = np.dot(x, model.weights)

        assert preds.min() >= 0
        if preds.max() < 256:
            preds = preds.astype('uint8')
        if '64' in str(xw.dtype):
            xw = xw.astype('float32')

        self.add_results(
          ['loss_indexed_image_classification',
           task.name,
           model.train_name,
           model.valid_name,
           use_history,
          ],
          {'erate': erate,
           'task_name': task.name,
           'train_name': model.train_name,
           'valid_name': model.valid_name,
           'use_history': use_history,
          },
          {
           'preds': preds,
           'xw': xw,
          })

        info('loss: ERR RATE %s = %f' % (task.name, erate))
        info('loss: XW STATS %f %f %f %s' %
                (xw.min(), xw.mean(), xw.max(), xw.shape))

        return erate


# -- this helper is called by mnist and svhn as well
def uslm_eval_helper(
    expr,
    memo,
    ctrl,
    data_fraction,
    assume_promising,
    data_view,
    memmap_name_template,
    DataView,
    loss_fn,
    true_loss_fn,
    ):

    use_obj_for_literal_in_memo(expr, data_view, DataView, memo)
    versions = git_versions()
    logger.info('GIT VERSIONS: %s' % str(versions))

    def exception_thrower():
        argdict = pyll.rec_eval(expr, memo=memo, print_node_on_error=False)
        visitor = PrimalVisitor(
            pipeline=argdict['pipeline'],
            ctrl=argdict['ctrl'],
            data_view=argdict['data_view'],
            max_n_features=argdict['max_n_features'],
            # TODO: just pass memmap_name directly
            memmap_name=memmap_name_template % (os.getpid(),
                                           np.random.randint(10000)),
            thresh_rank=1,
            optimize_l2_reg=True,
            batched_lmap_speed_thresh=argdict[
                'batched_lmap_speed_thresh'],
            badfit_thresh=None,
            batchsize=argdict['batchsize'],
            )

        protocol_iter = argdict['data_view'].protocol_iter(visitor)
        msg, model = protocol_iter.next()
        assert msg == 'model validation complete'

        # -- save the loss, but don't save attachments yet.
        rdict = visitor.hyperopt_rval()
        rdict['loss'] = loss_fn(visitor, argdict['bagging_fraction'])
        rdict['in_progress'] = True
        rdict['status'] = hyperopt.STATUS_OK
        argdict['ctrl'].checkpoint(rdict)

        if assume_promising:
            promising = True
        else:
            promising = view2_worth_calculating(
                loss=rdict['loss'],
                ctrl=argdict['ctrl'],
                thresh_loss=1.0,
                thresh_rank=1)

        logger.info('Promising: %s' % promising)
        if promising:
            msg, model2 = protocol_iter.next()
            assert msg == 'model testing complete'
            rdict = visitor.hyperopt_rval()
            rdict['loss'] = loss_fn(visitor, argdict['bagging_fraction'])
            rdict['true_loss'] = true_loss_fn(visitor)
            visitor.attach_obj_results()
        else:
            logger.warn('Not testing unpromising model %s' % str(model))
            del rdict['in_progress']
        return visitor, rdict

    try:
        visitor, rdict = call_catching_pipeline_errors(exception_thrower)
    except USLM_Exception, e:
        exc, rdict = e.args
        logger.info('job failed: %s: %s' % (type(e), exc))
    rdict['git_versions'] = versions
    return rdict

