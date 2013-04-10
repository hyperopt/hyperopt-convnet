import functools
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import numpy as np

from shovel import task

import hyperopt
from hyperopt import Trials
from hyperopt.mongoexp import MongoTrials

import hpconvnet.cifar10
import hpconvnet.slm


def make_trials(host, port, exp_key, refresh=True, dbname='dbname'):
    if (host, port) == (None, None):
        trials = Trials()
    else:
        if dbname == 'dbname':
            logger.warn('You probably want to override the default dbname')
        trials = MongoTrials(
                'mongo://%s:%d/%s/jobs' % (host, int(port), dbname),
                exp_key=exp_key,
                refresh=refresh)
    return trials


def dindex(d, *keys):
    """
    dindex(d, a, b, c) returns d[a][b][c]

    This function makes it easier to type long indexing sequences.
    """
    if keys:
        return dindex(d[keys[0]], *keys[1:])
    else:
        return d


@task
def small_random_run():
    # -- This is a smoke test to make sure that a lot of code paths actually
    # run. Some of the jobs will fail, some should succeed, the data will be
    # loaded and some SVMs will be fit etc. Classifier performance is expected
    # to be poor (70% error?), because we're using just 10% of the data and
    # only trying a few random architectures.
    #
    # Expected running time on CPU: ~10 mins

    search_space = hpconvnet.cifar10.build_search_space(
            max_n_features=4500,  # -- smaller than normal
            bagging_fraction=0.5, # -- normal
            n_unsup=2000, # -- smaller than normal
            abort_on_rows_larger_than=50 * 1000, # -- smaller
            )
    trials = Trials()
    hyperopt.fmin(
            fn=hyperopt.partial(
                hpconvnet.cifar10.uslm_eval,
                data_fraction=0.1, # -- smaller than normal
                ),
            space=search_space,
            algo=hyperopt.rand.suggest,
            max_evals=10,
            trials=trials)

    # -- Getting this far without crashing is a good indication that
    # everything has been installed and is probably running correctly.


@task
def tpe_driver(
    host,
    port, 
    max_evals=10000,
    exp_key_base='cifar10_tpe'
    ):

    max_evals=int(max_evals)
    min_ok_per_round = int(min_ok_per_round)

    tpe_suggest=functools.partial(
        hyperopt.tpe.suggest,
        n_startup_jobs=50, # -- number of random jobs before optimization
        )
    search_space = hpconvnet.cifar10.build_search_space(
            max_n_features=4000,
            bagging_fraction=1.0,
            n_unsup=7500, 
            abort_on_rows_larger_than=500 * 1000, # -- elements
            output_sizes=(32, 64),
            )
    hyperopt.fmin(
        fn=hpconvnet.cifar10.uslm_eval,
        space=search_space,
        algo=tpe_suggest,
        max_evals=max_evals,
        trials=make_trials(
            host,
            port,
            exp_key=exp_key,
            ),
        )

@task
def best_trial(host, port, exp_key='cifar10_tpe'):
    trials = make_trials(host, port, exp_key=exp_key)
    print 'Number of trials so far', len(trials)
    if len(trials) == 0:
        print ("(Hint: did you provide with the right exp_key? Used: %s)" %
            exp_key)

    try:
        best_trial = trials.best_trial
        print 'Best trial validation error rate', best_trial['result']['loss']
        print 'Best trial test error rate', best_trial['result']['true_loss']
    except ValueError:
        pass

