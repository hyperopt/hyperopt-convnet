"""
lfw.py - entry point for experiments on the LFW data set.


"""
import logging
import os

import numpy as np

import hyperopt
from hyperopt.base import use_obj_for_literal_in_memo
from hyperopt import STATUS_OK
from hyperopt import pyll

from skdata import lfw
from skdata import larray

from .pyll_slm import view2_worth_calculating
from .utils import git_versions

from .slm import call_catching_pipeline_errors
from .slm import USLM_Exception
from .slm import uslm_domain
from .slm_visitor_esvc import ESVC_SLM_Visitor
import slm_visitor_esvc

import foobar

warn = logging.getLogger(__name__).warn
info = logging.getLogger(__name__).info

# XXX: this is related to a hack for cacheing features to disk
#      see e.g. shovel/lfw.py, slm_visitor_esvc.py
dbname = 'lfw_db'


class DataViewPlaceHolder(object):
     pass

pyll_data_view = pyll.as_apply(DataViewPlaceHolder)


@pyll.scope.define
def unsup_images(data_view, trn, N):
    """
    Return a block of 
    """
    if trn == 'DevTrain':
        # -- extract training images, and put them into channel-major format
        imgs = larray.reindex(data_view.image_pixels,
                data_view.dev_train['lpathidx'][0, :N])[:]
        imgs = np.asarray(imgs)
        assert 'int' in str(imgs.dtype)
        foobar.append_ndarray_signature(imgs, 'unsup_images')
        foobar.append_trace('unsup_images N', N)
        return imgs.transpose(0, 3, 1, 2).copy()
    else:
        raise NotImplementedError()


def build_search_space(max_n_features, trn, n_unsup,
    bagging_fraction,
    batched_lmap_speed_thresh=None,
    batchsize=2,
    output_sizes=(32, 64, 128, 200),
    permit_affine_warp=True,
    ):
    image_shape = (250, 250, 1) # -- we're using lfw.Aligned below

    # max_n_features should be 16000 for full run
    # trn should be a string recognized by unsup_images()
    # n_unsup should be 300 for full run
    if batched_lmap_speed_thresh is None:
        batched_lmap_speed_thresh = {'seconds': 2.0, 'elements': 8}

    Xcm = pyll.scope.unsup_images(pyll_data_view, trn, n_unsup)
    search_space = {
            'data_view': pyll_data_view,
            'pipeline': uslm_domain(
                Xcm=Xcm,
                chmjr_image_shape=(
                    image_shape[2], image_shape[0], image_shape[1]),
                output_sizes=list(output_sizes), # -- is list required?
                batchsize=batchsize,
                max_n_features=max_n_features,
                batched_lmap_speed_thresh=batched_lmap_speed_thresh,
                permit_affine_warp=permit_affine_warp,
                ),
            'batchsize': batchsize,
            'max_n_features': max_n_features,
            'ctrl': hyperopt.Bandit.pyll_ctrl,
            'batched_lmap_speed_thresh': batched_lmap_speed_thresh,
            'bagging_fraction': bagging_fraction,
            }
    return search_space


@hyperopt.fmin_pass_expr_memo_ctrl
def slm_visitor_lfw(expr, memo, ctrl,
    maybe_test_view2=True,
    max_n_per_class=None,
    comparison_names=('mult', 'absdiff', 'sqrtabsdiff', 'sqdiff'),
    assume_promising=False,
    foobar_trace=True,
    foobar_trace_target=None,
    ):
    # -- possibly enable computation tracing
    foobar.reset_trace()
    foobar.trace_enabled = foobar_trace
    if foobar_trace_target:
        foobar.trace_verify = True
        foobar.set_trace_target(foobar_trace_target)
    slm_visitor_esvc._curdb = dbname # XXX tids are only unique within db

    versions = git_versions()
    info('GIT VERSIONS: %s' % str(versions))

    data_view = lfw.view.Aligned(
            x_dtype='uint8',
            max_n_per_class=max_n_per_class,
            )

    use_obj_for_literal_in_memo(expr, data_view, DataViewPlaceHolder, memo)

    def loss_fn(s, rdct, bagging_fraction):
        """
        bagging_fraction - float
            If the function measures the loss within the ensemble (loss)
            as well as the loss without the ensemble (loss_last_member) then
            this value interpolates between boosting (0.0) and bagging (1.0).

        """
        # -- this is the criterion we minimize during model search
        norm_key = s.norm_key('devTrain')
        task_name = 'devTrain'
        dct = s._results['train_image_match_indexed'][norm_key][task_name]
        loss = (bagging_fraction * dct['valid_error_no_ensemble']
                + (1 - bagging_fraction) * dct['valid_error'])
        rdct['loss'] = loss
        rdct['status'] = STATUS_OK

    def foo():
        argdict = pyll.rec_eval(expr, memo=memo, print_node_on_error=False)
        visitor = ESVC_SLM_Visitor(pipeline=argdict['pipeline'],
                    ctrl=argdict['ctrl'],
                    data_view=argdict['data_view'],
                    max_n_features=argdict['max_n_features'],
                    memmap_name='%s_%i' % (__name__, os.getpid()),
                    svm_crossvalid_max_evals=50,
                    optimize_l2_reg=True,
                    batched_lmap_speed_thresh=argdict[
                        'batched_lmap_speed_thresh'],
                    comparison_names=comparison_names,
                    batchsize=argdict['batchsize'],
                    )
        # -- drive the visitor according to the protocol of the data set
        protocol_iter = argdict['data_view'].protocol_iter(visitor)
        msg, model = protocol_iter.next()
        assert msg == 'model validation complete'

        # -- save the loss, but don't save attachments yet.
        rdict = visitor.hyperopt_rval(save_grams=False)
        rdict['in_progress'] = True
        loss_fn(visitor, rdict, argdict['bagging_fraction'])
        argdict['ctrl'].checkpoint(rdict)

        if assume_promising:
            promising = True
        else:
            promising = view2_worth_calculating(
                loss=rdict['loss'],
                ctrl=argdict['ctrl'],
                thresh_loss=1.0,
                thresh_rank=1)


        info('Promising: %s' % promising)

        if maybe_test_view2:
            if promising:
                info('Disabling trace verification for view2')
                foobar.trace_verify = False
                msg = protocol_iter.next()
                assert msg == 'model testing complete'
            else:
                warn('Not testing unpromising model %s' % str(model))
        else:
            warn('Skipping view2 stuff for model %s' % str(model))
        rdict = visitor.hyperopt_rval(save_grams=promising)
        loss_fn(visitor, rdict, argdict['bagging_fraction'])
        return visitor, rdict

    try:
        visitor, rdict = call_catching_pipeline_errors(foo)
    except USLM_Exception, e:
        exc, rdict = e.args
        print ('job failed: %s: %s' % (type(e), exc))
    rdict['git_versions'] = versions
    return dict(rdict, in_progres=False)

