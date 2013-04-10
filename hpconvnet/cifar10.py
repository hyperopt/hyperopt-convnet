import cPickle
import functools
import logging
import os

import numpy as np

from skdata.cifar10.views import StratifiedImageClassification

import hyperopt
from hyperopt import pyll

import pyll_slm  # adds the symbols to pyll.scope

from .slm_visitor_primal import uslm_eval_helper
from .slm import uslm_domain


dumps = functools.partial(cPickle.dumps, protocol=-1)
logger = logging.getLogger(__name__)


class DataView(object):
     pass

pyll_data_view = pyll.as_apply(DataView)


@pyll.scope.define
def cifar10_unsup_images(data_view, N):
    # -- extract training images for unsupervised learning,
    #    and put them into channel-major format
    imgs = np.asarray(
            data_view.dataset._pixels[
                data_view.fit_idxs[:N]])
    assert str(imgs.dtype) == 'uint8'
    rval = imgs.transpose(0, 3, 1, 2).copy()
    assert rval.shape[1] in (1, 3)  # -- channels
    return rval


def build_search_space(
    max_n_features,
    bagging_fraction,
    n_unsup,
    abort_on_rows_larger_than,
    batched_lmap_speed_thresh=None,
    batchsize=20,
    output_sizes=(32, 64, 128, 200),
    permit_affine_warp=True,
    ):
    if batched_lmap_speed_thresh is None:
        batched_lmap_speed_thresh = {'seconds': 2.0, 'elements': 150}
    Xcm = pyll.scope.cifar10_unsup_images(pyll_data_view, n_unsup)
    # -- currently these sizes are in *elements*
    search_space = {
            'data_view': pyll_data_view,
            'pipeline': uslm_domain(
                Xcm=Xcm,
                chmjr_image_shape=(3, 32, 32),
                output_sizes=output_sizes,
                batchsize=batchsize,
                max_n_features=max_n_features,
                batched_lmap_speed_thresh=batched_lmap_speed_thresh,
                permit_affine_warp=permit_affine_warp,
                abort_on_rows_larger_than=abort_on_rows_larger_than,
                ),
            'batchsize': batchsize,
            'max_n_features': max_n_features,
            'ctrl': hyperopt.Bandit.pyll_ctrl,
            'batched_lmap_speed_thresh': batched_lmap_speed_thresh,
            'bagging_fraction': bagging_fraction,
            }
    return search_space


def hybrid_loss(visitor, bagging_fraction):
    lossres = visitor._results['loss_indexed_image_classification']
    loss_ensemble = lossres['val']['fit']['val']['using_history']['erate']
    loss_member = lossres['val']['fit']['val']['not_using_history']['erate']
    loss = (bagging_fraction * loss_member
            + (1 - bagging_fraction) * loss_ensemble)
    return loss


def true_loss_fn(visitor):
    lossres = visitor._results['loss_indexed_image_classification']
    rval = lossres['tst']['sel']['None']['using_history']['erate']
    return rval


@hyperopt.fmin_pass_expr_memo_ctrl
def uslm_eval(
    expr, memo, ctrl,
    data_fraction=1.0,
    assume_promising=False,
    ):
    data_view = StratifiedImageClassification(
            dtype='uint8',
            n_train=int(40000 * data_fraction),
            n_valid=int(10000 * data_fraction),
            n_test=int(10000 * data_fraction),
            channel_major=False)

    memmap_name_template = 'cifar10_%i_%i'

    return uslm_eval_helper(expr, memo, ctrl, data_fraction, assume_promising,
                            data_view, memmap_name_template, DataView,
                            hybrid_loss, true_loss_fn)
