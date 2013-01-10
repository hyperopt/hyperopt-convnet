import os

import hyperopt
from hyperopt.base import use_obj_for_literal_in_memo
import numpy as np
from hyperopt import pyll
from skdata import lfw

from eccv12.slm import call_catching_pipeline_errors
from eccv12.slm import USLM_Exception
from eccv12.slm_visitor import SLM_Visitor

from eccv12.lfw import build_search_space
from eccv12.lfw import DataViewPlaceHolder

def test_one(rseed=3):
    # -- TEST: max_n_per_class < 1100
    data_view = lfw.view.Funneled(x_dtype='uint8', max_n_per_class=20)
    image_shape = data_view.image_pixels[0].shape
    assert image_shape == (250, 250, 3), image_shape

    expr = build_search_space(max_n_features=16000, trn='DevTrain',
            n_unsup=20, # -- TEST: small value here
            )
    expr = pyll.as_apply(expr)
    pyll.stochastic.recursive_set_rng_kwarg(expr, np.random.RandomState(rseed))
    ctrl = hyperopt.Ctrl(trials=hyperopt.Trials())
    memo = {hyperopt.Bandit.pyll_ctrl: ctrl}

    use_obj_for_literal_in_memo(expr, data_view, DataViewPlaceHolder, memo)

    def foo():
        argdict = pyll.rec_eval(expr, memo=memo, print_node_on_error=False)
        visitor = SLM_Visitor(pipeline=argdict['pipeline'],
                    ctrl=argdict['ctrl'],
                    data_view=argdict['data_view'],
                    max_n_features=argdict['max_n_features'],
                    memmap_name='%s_%i' % (__name__, os.getpid()),
                    thresh_loss=0.3,
                    thresh_rank=5,
                    loss_key=SLM_Visitor.test_key('devTrain', 'devTest'),
                    optimize_l2_reg=True,
                    )
        avg_test_score = argdict['data_view'].protocol(visitor)
        return avg_test_score, visitor
    try:
        avg_test_score, visitor = call_catching_pipeline_errors(foo)
        rdict = visitor.hyperopt_rval
    except USLM_Exception, e:
        exc, rdict = e.args
        print ('job failed: %s: %s' % (type(e), exc))
        return rdict

    print 'RESULT'
    print 'avg_test_score:', avg_test_score
    for k, v in sorted(rdict.items()):
        sv = str(v)
        if k == 'test_results':
            for kk, vv in sorted(v.items()):
                print '\t%s: %s' % (kk, vv)
        elif len(sv) > 80:
            print k, str(v)[:77] + '...'
        else:
            print k, sv
    return rdict


# TODO: mark this a SLOW test
def test_many(N=20):
    for rseed in range(1, N + 1):
        print 'test_many starting rseed:', rseed
        test_one(rseed)
        print 'test_many done rseed:', rseed
