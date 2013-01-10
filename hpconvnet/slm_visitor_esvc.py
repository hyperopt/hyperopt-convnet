import cPickle
import copy
import logging
import os
import time

from hyperopt.base import SONify

from .isvm_precomputed import EnsembleSVC
from .slm_visitor import SLM_Visitor
from .utils import loads_gram, dumps_gram
from .pyll_slm import error_rate
import foobar

import skdata.data_home

debug = logging.getLogger(__name__).debug
info = logging.getLogger(__name__).info
warn = logging.getLogger(__name__).warn

_curdb = 'curdb' # XXX: terrible hack :(
# _curdb is an abstraction leak -- MongoTrials has failed here.
# we set it from lfw.py


def cached_gram_load(tid, att_key):
    data_home = skdata.data_home.get_data_home()
    datafilename = os.path.join(data_home,
                 'hpconvnet', 'slm_visitor_esvc', _curdb, str(tid), att_key)
    return open(datafilename).read()


def cached_gram_save(tid, att_key, data):
    data_home = skdata.data_home.get_data_home()
    cachedir = os.path.join(data_home,
                 'hpconvnet', 'slm_visitor_esvc', _curdb, str(tid))
    datafilename = os.path.join(cachedir, att_key)
    info('Caching gram data %i/%s' % (tid, att_key))
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    datafile = open(datafilename, 'w+')
    datafile.write(data)
    datafile.close()


class ESVC_SLM_Visitor(SLM_Visitor):
    """
    Use an EnsembleSVC classifier, suitable for datasets with not too many
    examples (< 20000) and binary labels.
    """
    def __init__(self,
            optimize_l2_reg=False,
            svm_crossvalid_max_evals=20,
            **kwargs):
        SLM_Visitor.__init__(self, **kwargs)
        self.optimize_l2_reg = optimize_l2_reg
        self.member_name = self._member_name()
        self.svm_crossvalid_max_evals = svm_crossvalid_max_evals

        self._results = {
            'train_image_match_indexed': {},
            'retrain_classifier_image_match_indexed': {},
            'loss_image_match_indexed': {},
        }

        if not self.optimize_l2_reg:
            raise NotImplementedError()

    def norm_key(self, sample, tid=None):
        if tid is None:
            member_name = self.member_name
        else:
            member_name = self._member_name(tid)
        norm_key = 'nkey_%s_%s' % (member_name, sample)
        return norm_key

    def load_ensemble_weights(self, norm_sample, task_name, ens):
        # -- load the weights from the most recent ensemble, if there is one.
        for trial in self.history[-1:]:
            info('Loading weights from document %i' % trial['tid'])
            trial_norm_key = self.norm_key(norm_sample, tid=trial['tid'])
            trial_weights = trial['result']['weights']
            norm_task_weights = trial_weights[trial_norm_key][task_name]
            for norm_key, weight in norm_task_weights.items():
                if ens.has_member(norm_key):
                    ens.set_weight(norm_key, weight)
                else:
                    ens.add_member(norm_key, weight)
                info(' .. weight[%s] = %s' % (norm_key, weight))
                foobar.append_trace('load ensemble weights', norm_key, weight)

    def load_ensemble_grams(self, norm_sample, ens, sample1, sample2):
        trial_attachments = self.ctrl.trials.trial_attachments

        # -- load the gram matrices saved by each ensemble member
        for trial in self.history:
            trial_norm_key = self.norm_key(norm_sample, tid=trial['tid'])
            info('Loading grams from document %i' % trial['tid'])
            debug(' .. saved_grams: %s' %
                    str(trial['result']['grams'][trial_norm_key]))
            for (s1, s2) in trial['result']['grams'][trial_norm_key]:
                if set([sample1, sample2]) == set([s1, s2]):
                    if not ens.has_gram(trial_norm_key, s1, s2):
                        att_key = 'gram_%s_%s_%s.pkl' % (trial_norm_key, s1, s2)
                        info('retrieving gram_data %i:%s'
                             % (trial['tid'], att_key))
                        try:
                            gram_data = cached_gram_load(trial['tid'], att_key)
                        except IOError:
                            gram_data = trial_attachments(trial)[att_key]
                            cached_gram_save(trial['tid'], att_key, gram_data)
                        info('retrieved %i bytes' % len(gram_data))
                        gram = loads_gram(gram_data)
                        if s1 == sample1:
                            ens.add_gram(trial_norm_key, sample1, sample2, gram)
                        else:
                            ens.add_gram(trial_norm_key, sample1, sample2,
                                    gram.T)
                        foobar.append_ndarray_signature(
                            gram,
                            'load gram', trial_norm_key, sample1, sample2)
            info('Loading grams done')

    def hyperopt_rval(self, save_grams):
        rval = copy.deepcopy(self._results)
        rval['attachments'] = {}
        rval['grams'] = {}
        rval['weights'] = {}
        rval['trace'] = copy.deepcopy(foobar._trace)

        saved = set()

        def jsonify_train_results(rkey):
            for norm_key in rval[rkey]:

                for task_name in rval[rkey][norm_key]:
                    svm_dct = rval[rkey][norm_key][task_name]
                    ens = svm_dct.pop('ens')

                    rval['weights'].setdefault(norm_key, {})
                    rval['weights'][norm_key][task_name] = ens._weights

                    # -- stash these as attachments because they fill up the db.
                    xmean = svm_dct.pop('xmean')
                    xstd = svm_dct.pop('xstd')

                    if save_grams:
                        xmean_key = 'xmean_%s_%s_%s' % (rkey, norm_key, task_name)
                        xstd_key = 'xstd_%s_%s_%s' % (rkey, norm_key, task_name)
                        rval['attachments'][xmean_key] = cPickle.dumps(xmean, -1)
                        rval['attachments'][xstd_key] = cPickle.dumps(xstd, -1)

                        rval['grams'].setdefault(norm_key, [])
                        for (inorm_key, sample1, sample2) in ens._grams:
                            if inorm_key != norm_key:
                                # -- we're only interested in saving the grams
                                # calculated by this run.
                                continue
                            if (norm_key, sample1, sample2) in saved:
                                # -- already saved this one
                                continue

                            att_key = 'gram_%s_%s_%s.pkl' % (
                                    norm_key, sample1, sample2)

                            info('saving %s' % att_key)

                            gram = ens._grams[(norm_key, sample1, sample2)]
                            rval['attachments'][att_key] = dumps_gram(
                                gram.astype('float32'))

                            rval['grams'][norm_key].append((sample1, sample2))

                            saved.add((norm_key, sample1, sample2))
                            saved.add((norm_key, sample2, sample1))

        jsonify_train_results('train_image_match_indexed')
        jsonify_train_results('retrain_classifier_image_match_indexed')

        return SONify(rval)

    def forget_task(self, task_name):

        # free up RAM by deleting all features computed for task_name
        def delete_features(rkey):
            for norm_key in self._results[rkey]:
                if task_name in self._results[rkey][norm_key]:
                    svm_dct = self._results[rkey][norm_key][task_name]
                    svm_dct['ens'].del_features(norm_key, task_name)

        delete_features('train_image_match_indexed')
        delete_features('retrain_classifier_image_match_indexed')

    def train_image_match_indexed(self, task, valid=None):

        pipeline = self.pipeline

        info('training svm on %s' % task.name)
        ens = EnsembleSVC(task.name)

        norm_task = task.name
        norm_key = self.norm_key(norm_task)
        svm_dct = {
                'ens': ens,
                'norm_key': norm_key,
                'norm_task': task.name,
                'task_name': task.name,
                }

        ens.add_member(norm_key)
        ens.add_sample(task.name, task.y)
        x_trn = self.normalized_image_match_features(task, svm_dct,
                role='train')
        ens.add_features(norm_key, task.name, x_trn)

        foobar.append_ndarray_signature(x_trn,
            'train_image x_trn', norm_key, task.name)

        info('computing gram: %s / %s / %s' % (
            norm_key, task.name, task.name))
        ens.compute_gram(norm_key, task.name, task.name, dtype='float32')

        foobar.append_ndarray_signature(
            ens._grams[(norm_key, task.name, task.name)],
            'train_image train_gram', norm_key, task.name)

        if valid is not None:
            info('cross-validating svm on %s' % valid.name)
            x_val = self.normalized_image_match_features(valid, svm_dct,
                    role='test',
                    # -- assume that slow features were caught earlier
                    batched_lmap_speed_thresh={'seconds': 30, 'elements': 1},
                    )
            foobar.append_ndarray_signature(
                x_val,
                'train_image x_val', norm_key, valid.name, task.name)

            ens.add_sample(valid.name, valid.y)
            ens.add_features(norm_key, valid.name, x_val)

            info('computing gram: %s / %s / %s' % (
                norm_key, valid.name, task.name))
            ens.compute_gram(norm_key, valid.name, task.name, dtype='float32')
            foobar.append_ndarray_signature(
                ens._grams[(norm_key, valid.name, task.name)],
                'train_image valid_gram', norm_key, valid.name, task.name)

            # -- re-fit the model using best weights on train + valid sets
            info('computing gram: %s / %s / %s' % (
                norm_key, valid.name, valid.name))
            ens.compute_gram(norm_key, valid.name, valid.name, dtype='float32')

            train_valid = '%s_%s' % (task.name, valid.name)
            ens.add_compound_sample(train_valid, [task.name, valid.name])


        def load_history():
            info('loading history')
            self.load_ensemble_history(
                fields=['result.weights','result.grams'])
            self.load_ensemble_weights(norm_task, task.name, ens)
            self.load_ensemble_grams(norm_task, ens, task.name, task.name)
            if valid is not None:
                self.load_ensemble_grams(norm_task, ens, valid.name, task.name)
                self.load_ensemble_grams(norm_task, ens, valid.name, valid.name)


        def train_main():
            ens.train_sample = task.name

            t0 = time.time()
            if valid is None:
                svm_dct['l2_reg'] = pipeline['l2_reg']
                ens.fit_svm(svm_dct['l2_reg'])
                svm_dct['train_error'] = ens.error_rate(task.name)
                svm_dct['loss'] = svm_dct['train_error']
            else:

                #scales = {m: 3.0 for m in ens._weights}
                scales = dict([(m, 3.0) for m in ens._weights])
                scales[norm_key] = 100.0

                info('fit_weights_crossvalid(%s, %i)' % (
                    valid.name, self.svm_crossvalid_max_evals))
                ens.fit_weights_crossvalid(valid.name,
                        max_evals=self.svm_crossvalid_max_evals,
                        scales=scales)

                foobar.append_trace('xvalid weights', sorted(ens._weights.items()))

                svm_dct['task_error'] = ens.error_rate(task.name)
                foobar.append_trace('task_error', svm_dct['task_error'])

                svm_dct['valid_name'] = valid.name
                svm_dct['valid_error'] = ens.error_rate(valid.name)
                info('valid_error %f' % svm_dct['valid_error'])
                foobar.append_trace('valid_error', svm_dct['valid_error'])

                svm_dct['l2_reg'] = None  # -- use default when retraining

                # -- re-fit the model using best weights on train + valid sets
                ens.train_sample = train_valid
                ens.fit_svm()

            fit_time = time.time() - t0
            svm_dct['fit_time'] = fit_time


        info('training with just the current features...')
        train_main()
        svm_dct['task_error_no_ensemble'] = svm_dct['task_error']
        svm_dct['valid_error_no_ensemble'] = svm_dct['valid_error']

        load_history()
        if self.history:
            info('training the full ensemble...')
            train_main()

        try:
            print_summary = ens.print_summary
        except AttributeError:
            print_summary = lambda : None

        print_summary()

        dct = self._results['train_image_match_indexed']
        dct.setdefault(norm_key, {})
        if task.name in dct[norm_key]:
            warn('Overwriting train_image_match_indexed result: %s'
                 % task.name)
        dct[norm_key][task.name] = svm_dct

        return svm_dct

    def retrain_classifier_image_match_indexed(self, model, task):
        # We are making the decision that retraining a classifier means not
        # retraining the weights or the features, but just retraining the
        # libsvm part.

        ens = model['ens'].copy()
        ens.train_sample = task.name
        svm_dct = dict(
                ens=ens,
                norm_key=model['norm_key'],
                norm_task=model['norm_task'],
                task_name=task.name,
                xmean=model['xmean'],
                xstd=model['xstd'],
                l2_reg=model['l2_reg'],
                )
        if 'divrowl2_avg_nrm' in model:
            svm_dct['divrowl2_avg_nrm'] = model['divrowl2_avg_nrm']
        norm_key = svm_dct['norm_key']
        norm_task = svm_dct['norm_task']
        info('retraining on %s (norm_task=%s)' % (task.name, norm_task))

        ens.add_sample(task.name, task.y)
        x_trn = self.normalized_image_match_features(task, svm_dct,
                # -- do not recompute mean and var
                role='test',
                # -- assume that slow features were caught earlier
                batched_lmap_speed_thresh={'seconds': 30, 'elements': 1},
                )
        ens.add_features(norm_key, task.name, x_trn)

        self.load_ensemble_grams(norm_task, ens, task.name, task.name)
        ens.compute_gram(norm_key, task.name, task.name, dtype='float32')

        ens.fit_svm(svm_dct['l2_reg'])
        svm_dct['task_error'] = ens.error_rate(task.name)

        info('retrain_classifier: %s -> %f' % (
            (norm_key, task.name), svm_dct['task_error']))

        dct = self._results['retrain_classifier_image_match_indexed']
        dct.setdefault(norm_key, {})
        if task.name in dct[norm_key]:
            warn('Overwriting retrain_classifier_image_match_indexed result: %s'
                 % task.name)
        dct[norm_key][task.name] = svm_dct
        return svm_dct

    def loss_image_match_indexed(self, svm_dct, task):
        norm_task = svm_dct['norm_task']
        norm_key = svm_dct['norm_key']

        info('loss_image_match_indexed: %s, %s' % (norm_key, task.name) )
        x = self.normalized_image_match_features(task, svm_dct, 'test',
                # -- assume that slow features were caught earlier
                batched_lmap_speed_thresh={'seconds': 30, 'elements': 1},
                )
        svm_dct['ens'].add_sample(task.name, task.y)
        svm_dct['ens'].add_features(norm_key, task.name, x)

        self.load_ensemble_grams(norm_task, svm_dct['ens'], task.name,
                svm_dct['ens'].train_sample)
        svm_dct['ens'].compute_gram(norm_key, task.name,
                svm_dct['ens'].train_sample, dtype='float32')

        preds = svm_dct['ens'].predict(task.name)
        erate = error_rate(preds, task.y)
        info('test_image_match_indexed error_rate %s -> %f' % (
            task.name, erate))

        # -- add summary information to self._results
        dct = self._results['loss_image_match_indexed']
        dct.setdefault(norm_key, {})
        if task.name in dct[norm_key]:
            warn('Overwriting loss_image_match_indexed result: %s'
                 % task.name)
        dct[norm_key][task.name] = {
            'error_rate': erate,
            'norm_key': norm_key,
            'task_name': task.name,
            'preds_01': ''.join(
                ['0' if p == -1 else '1' for p in preds]),
            }
        return erate

