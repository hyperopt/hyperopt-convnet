import functools
import logging
import time

import numpy as np

import skdata.base
from skdata.larray import cache_memmap
from skdata.larray import lmap
from skdata.larray import lmap_info
from skdata.larray import reindex

from hyperopt.pyll import scope # TODO remove-me
import foobar

from .pyll_slm import average_row_l2norm
from .pyll_slm import pyll_theano_batched_lmap # TODO: CLEAN UP
from .pyll_slm import flatten_elems

from .isvm_boosting import BoostHelper

from .utils import mean_and_std
import comparisons

info = logging.getLogger(__name__).info
warn = logging.getLogger(__name__).warn


class SLM_Visitor(skdata.base.LearningAlgo):
    """
    This class takes an *evaluated* pipeline as a starting point,
    and applies it to a data set.  Nothing symbolic here.
    """
    def __init__(self, pipeline, ctrl, data_view,
        max_n_features,
        batchsize,
        memmap_name='',
        badfit_thresh=float('inf'),
        memmap_del_atexit=True,
        batched_lmap_speed_thresh=None,
        comparison_names=('mult', 'absdiff', 'sqrtabsdiff', 'sqdiff'),
        ):
        self.pipeline = pipeline
        self.ctrl = ctrl
        self.data_view = data_view
        self.memmap_name = memmap_name
        self.max_n_features = max_n_features
        self.badfit_thresh = badfit_thresh
        self.memmap_del_atexit = memmap_del_atexit
        self.batched_lmap_speed_thresh = batched_lmap_speed_thresh
        self.batchsize = batchsize

        self.image_features = {}
        self.comparison_names = comparison_names

    def get_image_features_lmap(self, images, batched_lmap_speed_thresh=None):
        N, H, W, C = images.shape
        assert C in (1, 3)
        # -- this loading must be simple, and match the unsup_images
        # function in lfw. Anything more elaborate must
        # be included in the pyll pipeline
        chmajor_fn = functools.partial(np.transpose, axes=(2, 0, 1))
        chmajor_fn = lmap_info(
            shape=(C, H, W),
            dtype=images.dtype
            )(chmajor_fn)
        def chmajor_fn_f_map(X):
            return np.transpose(X, axes=(0, 3, 1, 2))
        chmajor_fn.f_map = chmajor_fn_f_map

        rval = pyll_theano_batched_lmap(
                scope.partial(scope.callpipe1, self.pipeline['pipe']),
                lmap(chmajor_fn, images),
                batchsize=self.batchsize,
                print_progress_every=10,  # -- seconds
                abort_on_rows_larger_than=self.max_n_features,
                speed_thresh=batched_lmap_speed_thresh,
                x_dtype='uint8', # HAS TO MATCH ./slm.py
                )
        return rval

    # XXX ugly function, refactor with normalized_image_match_features
    #     we don't even use the "cdict" anymore and I think I manually
    #     clear the self.image_features dict after calling!
    def get_image_features(self, task, batched_lmap_speed_thresh=None):
        if batched_lmap_speed_thresh is None:
            batched_lmap_speed_thresh = self.batched_lmap_speed_thresh
        images = task.images
        try:
            rval, _images, cdict = self.image_features[images]
            # -- currently it is true that all tasks should be indexing into
            # -- the same set of images. Later when this is not the case,
            # -- delete this over-strict check.
            assert _images is images
        except KeyError:
            feature_lmap = self.get_image_features_lmap(task.images,
                    batched_lmap_speed_thresh)

            rval = cache_memmap(
                feature_lmap,
                self.memmap_name + '_image_features_' + task.name,
                del_atexit=self.memmap_del_atexit)

            foobar.append_ndarray_signature(rval[0],
                'get_image_features features 0', task.name)
            foobar.append_ndarray_signature(rval[100],
                'get_image_features features 100', task.name)

            cdict = {}
            self.image_features[images] = rval, images, cdict
        return rval, cdict

    def normalized_image_features(self, images, xmean, xstd, avg_nrm,
        n_rows_to_estimate_stats=1000,
        flatten=True,
        batched_lmap_speed_thresh=None,
        ):
        """
        svm_dct - dict
            dict of parameters for normalization:
                'remove_std0'
                'varthresh'
                'divrowl2'
            write xmean, xstd if role is 'train'
            read xmean and xstd if role is 'test'
        role - str
            either 'train' or 'test'
        n_rows_to_estimate_stats - bool
            estimate xmean and xstd from the first N feature vectors
        flatten - bool
            return features flattened to vectors
        """

        if not flatten:
            raise NotImplementedError('only flatten is implemented')

        pipeline = self.pipeline
        features_lmap = self.get_image_features_lmap(images)

        n_features = np.prod(features_lmap.shape[1:])

        if xmean is None:
            # -- load enough training data into memory to estimate stats
            cache_train = flatten_elems(
                features_lmap[:n_rows_to_estimate_stats])

            xmean, xstd = mean_and_std(
                cache_train,
                remove_std0=pipeline['remove_std0'])

            xstd = np.sqrt(xstd ** 2 + pipeline['varthresh'])

            if pipeline['divrowl2']:
                avg_nrm = 1e-7 + average_row_l2norm(
                    (cache_train - xmean) / xstd)
            else:
                avg_nrm = 1

        def normalize(x):
            return (x.flatten() - xmean) / (xstd * avg_nrm)

        def normalize_many(x):
            return (x.reshape((len(x), -1)) - xmean) / (xstd * avg_nrm)

        normed_features = lmap(
            lmap_info(
                shape=(n_features,),
                dtype=features_lmap.dtype)(normalize),
            features_lmap,
            ragged=False,
            f_map=normalize_many)

        return normed_features, xmean, xstd, avg_nrm



    def normalized_image_match_features(self, task, svm_dct, role,
            batched_lmap_speed_thresh=None):
        assert role in ('train', 'test')
        if batched_lmap_speed_thresh is None:
            batched_lmap_speed_thresh = self.batched_lmap_speed_thresh
        image_features, cdict = self.get_image_features(task,
                batched_lmap_speed_thresh=batched_lmap_speed_thresh)
        del cdict # -- no longer used (waste of memory)
        pipeline = self.pipeline
        info('Indexing into image_features of shape %s' %
                str(image_features.shape))

        comps = [getattr(comparisons, cc)
                for cc in self.comparison_names]
        n_features = np.prod(image_features.shape[1:])
        n_trn = len(task.lidx)

        x_trn_shp = (n_trn, len(comps), n_features)
        info('Allocating training ndarray of shape %s' % str(x_trn_shp))
        x_trn = np.empty(x_trn_shp, dtype='float32')

        # -- pre-compute all of the image_features we will need
        all_l_features = reindex(image_features, task.lidx)[:]
        all_r_features = reindex(image_features, task.ridx)[:]

        all_l_features = all_l_features.reshape(len(all_l_features), -1)
        all_r_features = all_r_features.reshape(len(all_r_features), -1)

        foobar.append_ndarray_signature(all_l_features,
            'normalized_image_match l_features', task.name)
        foobar.append_ndarray_signature(all_r_features,
            'normalized_image_match r_features', task.name)

        if role == 'train':
            if np.allclose(all_l_features.var(axis=0), 0.0):
                raise ValueError(
                    'Homogeneous features (non-finite features)')

            xmean_l, xstd_l = mean_and_std(all_l_features,
                    remove_std0=pipeline['remove_std0'])
            xmean_r, xstd_r = mean_and_std(all_r_features,
                    remove_std0=pipeline['remove_std0'])
            xmean = (xmean_l + xmean_r) / 2.0
            # -- this is an ad-hoc way of blending the variances.
            xstd = np.sqrt(np.maximum(xstd_l, xstd_r) ** 2
                           + pipeline['varthresh'])

            foobar.append_ndarray_signature(
                xmean, 'normalized_image_match xmean', task.name)
            foobar.append_ndarray_signature(
                xstd, 'normalized_image_match xstd', task.name)


            svm_dct['xmean'] = xmean
            svm_dct['xstd'] = xstd
        else:
            xmean = svm_dct['xmean']
            xstd = svm_dct['xstd']

        info('Computing comparison features')

        # -- now compute the "comparison functions" into x_trn
        for jj, (lfeat, rfeat) in enumerate(
                zip(all_l_features, all_r_features)):
            lfeat_z = (lfeat - xmean) / xstd
            rfeat_z = (rfeat - xmean) / xstd
            for ci, comp in enumerate(comps):
                x_trn[jj, ci, :] = comp(lfeat_z, rfeat_z)

        if pipeline['divrowl2']:
            info('Dividing by feature norms')
            # -- now normalize by average feature norm because some
            #    comparison functions come out smaller than others
            if role == 'train':
                svm_dct['divrowl2_avg_nrm'] = {}
                for ci, cname in enumerate(self.comparison_names):
                    avg_nrm = average_row_l2norm(x_trn[:, ci, :]) + 1e-7
                    svm_dct['divrowl2_avg_nrm'][cname] = avg_nrm

            avg_nrm_vec = [svm_dct['divrowl2_avg_nrm'][cname]
                           for cname in self.comparison_names]
            x_trn /= np.asarray(avg_nrm_vec)[None, :, None]
            foobar.append_trace('get_normlized_features avg_nrm', avg_nrm_vec)

        # -- collapse comparison and feature dimensions
        x_trn.shape = (x_trn.shape[0], x_trn.shape[1] * x_trn.shape[2])

        foobar.append_ndarray_signature(
            x_trn, 'normalized_image_match x_trn', task.name)
        info('normalized_image_match_features complete')
        return x_trn

    def loss(self, model, task):
        info('Score %s' % task.name)
        semantics = task.semantics
        methodname = 'loss_' + semantics
        method = getattr(self, methodname)
        loss = method(model, task)
        return loss

    def best_model(self, train, valid=None):
        semantics = train.semantics
        # -- train the svm
        info('BestModelByCrossValidation %s, %s' % (
            train.name, getattr(valid, 'name', None)))
        model = getattr(self, 'train_' + semantics)(train, valid)
        return model

    def retrain_classifier(self, model, task):
        info('RetrainClassifier %s' % task.name)
        semantics = task.semantics
        methodname = 'retrain_classifier_' + semantics
        method = getattr(self, methodname)
        new_model = method(model, task)
        # -- measure the erate and compute the cur_xw values
        getattr(self, 'loss_' + semantics)(new_model, task)
        return new_model

    def _member_name(self, tid=None):
        if tid is None:
            if self.ctrl.current_trial is None:
                tid = 'debug'
            else:
                tid = self.ctrl.current_trial['tid']
        member_name = 'member_%s' % tid
        return member_name

    def load_ensemble_history(self, fields):

        trials = self.ctrl.trials
        if hasattr(trials, 'handle'):
            # query mongodb directly to avoid transferring un-necessary fields
            docs_for_bh = BoostHelper.query_MongoTrials(
                trials,
                fields=fields)
            # download only those docs that are in the active history
            trials.refresh_tids([d['tid'] for d in docs_for_bh])
            # -- XXX: relatively arbitrary assert to make sure we didn't
            # download a whole wack of documents... the point of
            # refresh_tids is to avoid this.
            assert len(trials.trials) < len(docs_for_bh) + 5, (
                len(trials.trials), len(docs_for_bh))
        else:
            trials.refresh()
            docs_for_bh = trials.trials

        def helper():
            bh = BoostHelper(docs_for_bh)

            if self.ctrl.current_trial is None:
                history = []
            else:
                history = bh.history(self.ctrl.current_trial)
                assert history[-1] is self.ctrl.current_trial
                history.pop(-1)
            info('load_ensemble_history: %i previous model documents found'
                    % len(history))
            return history

        retries = 20
        while retries:
            history = helper()
            if any(trial['result'].get('in_progress') for trial in history):
                warn('Previous trial is still in progress, waiting 30s')
                time.sleep(30)
                retries -= 1
            else:
                break

        foobar.append_trace('load ensemble history len', len(history))

        if retries:
            self.history = history
        else:
            raise Exception('Previous trial in progress, cannot continue')

