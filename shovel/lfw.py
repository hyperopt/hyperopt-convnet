"""
Driver scripts for LFW experiments

"""

import numpy as np
from functools import partial

from shovel import task

import hyperopt
from hyperopt import Trials
from hyperopt.mongoexp import MongoTrials, MongoCtrl

from hpconvnet.shovel_util import configure_logging
import hpconvnet.lfw

configure_logging('INFO')
dbname = hpconvnet.lfw.dbname


def make_trials(host, port, exp_key, refresh=True):
    if (host, port) == (None, None):
        trials = Trials()
    else:
        trials = MongoTrials(
                'mongo://%s:%d/%s/jobs' % (host, int(port), dbname),
                exp_key=exp_key,
                refresh=refresh)
    return trials


search_space = partial(
        hpconvnet.lfw.build_search_space,
        max_n_features=16000,
        trn='DevTrain', # -- split used for unsupervised images
        n_unsup=300,    # -- number of images from which to draw patches
        )


def slm_visitor_lfw_partial(
    max_n_per_class,
    maybe_test_view2=False, # -- this still takes too much memory
    assume_promising=False,
    foobar_trace_target=None,
    ):
    # -- this curries and re-decorates hpconvnet.lfw.slm_visitor_lfw
    #    so that we can pass it to fmin()
    if max_n_per_class is not None:
        max_n_per_class = int(max_n_per_class)
    return hyperopt.partial(
        hpconvnet.lfw.slm_visitor_lfw,
        max_n_per_class=max_n_per_class,
        maybe_test_view2=maybe_test_view2,
        assume_promising=assume_promising,
        foobar_trace_target=foobar_trace_target,
        )


@task
def random_driver(host=None, port=None, max_evals=10000, max_n_per_class=None):
    hyperopt.fmin(
            fn=slm_visitor_lfw_partial(max_n_per_class),
            space=search_space(bagging_fraction=1.0),
            algo=hyperopt.rand.suggest,
            max_evals=max_evals,
            trials=make_trials(host, port, exp_key='random'))


@task
def tpe_driver(host=None, port=None, max_evals=10000, max_n_per_class=None):
    tpe_suggest=partial(
        hyperopt.tpe.suggest,
        n_startup_jobs=50, # -- number of random jobs before optimization
        )
    hyperopt.fmin(
            fn=slm_visitor_lfw_partial(max_n_per_class),
            space=search_space(bagging_fraction=1.0),
            algo=tpe_suggest,
            max_evals=max_evals,
            trials=make_trials(host, port, exp_key='random'))


@task
def view2(host, port, exp_key,
    bagging_fraction,
    max_n_per_class=None,
    maybe_test_view2=True,
    assume_promising=True,
    tid=None,
    fake=False,
    ):
    fake = int(fake)
    real_trials = make_trials(host, port, exp_key)
    print 'n. real trials', len(real_trials)
    if tid is None:
        best_trial = real_trials.best_trial
    else:
        try:
            best_trial = [t for t in real_trials.trials if t['tid'] == int(tid)][0]
        except IndexError:
            print [t['tid'] for t in real_trials.trials]
    print 'Best trial'
    print ' ["tid"]', best_trial['tid']
    best_result = best_trial['result']
    print 'Best trial had loss', best_result['loss']
    best_trace = best_result.get('trace')
    if 0:
        print ' ["Result trace"]', best_trace
    fake_trials = hyperopt.Trials()
    fn = slm_visitor_lfw_partial(
        max_n_per_class,
        maybe_test_view2=int(maybe_test_view2),
        assume_promising=int(assume_promising),
        foobar_trace_target=list(best_trace),
        )
    #space = search_space
    space = hpconvnet.lfw.build_search_space(
            max_n_features=16000,
            trn='DevTrain', # -- split used for unsupervised images
            n_unsup=300,    # -- number of images from which to draw patches
            batched_lmap_speed_thresh={'seconds': 60, 'elements': 1},
            bagging_fraction=float(bagging_fraction),
            )
    domain = hyperopt.Domain(fn, space, rseed=123)
    domain.rng = None  # -- this rng is never to be used
    if fake:
        ctrl = hyperopt.Ctrl(trials=fake_trials, current_trial=None)
        print 'WARNING: running on fake ctrl object'
    else:
        ctrl = MongoCtrl(
            trials=real_trials,
            current_trial=best_trial,
            read_only=False)

    #real_trials.handle.update(best_result, msg)
    #ctrl.checkpoint(dict(best_trial['result'], foodebug='yes'))
    config = hyperopt.base.spec_from_misc(best_trial['misc'])
    #print 'Config', config
    r_dct = domain.evaluate(config, ctrl, attach_attachments=(not fake))
    print 'r_dct'
    print r_dct
    if fake:
        print 'WARNING: running on fake ctrl object, not saving result'
        attachments = r_dct.pop('attachments', {})
        print 'Attachments:', attachments.keys()
        print ' ["Best Result trace"]'
        def print_trace(r):
            trace = r['trace']
            for t in trace:
                print '  ', t
        print_trace(best_result)
        print ' ["result trace"]'
        print_trace(r_dct)
    else:
        # -- the loss should have been re-computed identically
        r_dct['view2_recalculated_loss'] = r_dct['loss']
        r_dct['loss'] = best_result['loss']
        if  (r_dct['view2_recalculated_loss'] > best_result['loss']):
            print 'WARNING: recalculated loss was worst than loss during search'
            print ' -> original loss', best_result['loss']
            print ' -> recalculated loss', r_dct['view2_recalculated_loss']

        print 'Checkpointing back to db'
        # -- N.B. attachments should have been saved by Domain.evaluate,
        #    since we called it with attach_attachments=True. So they should 
        #    not be here anymore.
        assert 'attachments' not in r_dct

        ctrl.checkpoint(r_dct)

@task
def list_view2_scores(host, port, key):
    """
    List any view2 scores that have been calculated within an experiment
    (exp_key)
    """
    import pymongo as pm
    conn = pm.Connection(host=host, port=int(port))
    query = {'exp_key': key}

    for doc in conn[dbname]['jobs'].find(query, fields=['tid', 'result', 'state']):
        r = doc['result'].get('loss_image_match_indexed')
        if r:
            assert len(r) == 1
            for member in r:
                fold_errs = []
                for view_train_k in r[member]:
                    if 'test' in view_train_k:
                        fold_errs.append(r[member][view_train_k]['error_rate'])
            print list(enumerate(fold_errs))
            print key, doc['tid'],
            print 'view1', doc['result']['loss'],
            print 'view2', np.mean(fold_errs),
            print doc['state'], doc['result']['status']

