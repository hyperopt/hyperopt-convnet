import hashlib
import time
import numpy as np
import StringIO
import traceback

from hyperopt import pyll
from hyperopt.pyll import scope

import hyperopt
from hyperopt.pyll_utils import hp_choice
from hyperopt.pyll_utils import hp_uniform
from hyperopt.pyll_utils import hp_quniform
from hyperopt.pyll_utils import hp_loguniform
from hyperopt.pyll_utils import hp_qloguniform
from hyperopt.pyll_utils import hp_normal
from hyperopt.pyll_utils import hp_lognormal

import pyll_slm  # adds the symbols to pyll.scope

pyll.scope.import_(globals(),
    # -- from pyll
    'partial',
    'callpipe1',
    'switch',
    'sqrt',
    #
    # -- misc. from ./pyll_slm.py
    'pyll_theano_batched_lmap',
    'model_predict',
    'model_decisions',
    'error_rate',
    'mean_and_std',
    'flatten_elems',
    'np_transpose',
    'np_RandomState',
    'print_ndarray_summary',
    'pickle_dumps',
    #
    # -- filterbank allocators  (./pyll.slm.py)
    'random_patches',
    'alloc_random_uniform_filterbank',
    'patch_whitening_filterbank_X',
    'fb_whitened_patches',
    'fb_whitened_projections',
    'slm_uniform_M_FB',
    #
    # -- pipeline elements  (./pyll.slm.py)
    'slm_affine_image_warp',
    'slm_img_uint8_to_float32',
    'slm_lpool',
    'slm_lnorm',
    'slm_lpool_alpha',
    'slm_fbncc_chmaj',
    'slm_quantize_gridpool',
    #
    # -- renamed symbols
    **{
    # NEW NAME:         ORIG NAME
    's_int':           'int',
    's_float':         'float',
    'pyll_getattr':    'getattr',
    })

# -- where is this supposed to go?
divide_by_avg_norm=False


def stable_hash(s):
    if isinstance(s, basestring):
        return int(hashlib.sha224(s).hexdigest(), 16)
    else:
        raise TypeError(s)


def hp_TF(label):
    return hp_choice(label, [0, 1])


def rfilter_size(label, smin, smax, q=1):
    """Return an integer size from smin to smax inclusive with equal prob
    """
    return s_int(hp_quniform(label, smin - q / 2.0 + 1e-5, smax + q / 2.0, q))


def logu_range(label, lower, upper):
    """Return a continuous replacement for one_of(.1, 1, 10)"""
    return hp_loguniform(label, np.log(lower), np.log(upper))


def new_fbncc_layer(prefix, Xcm, n_patches, n_filters, size,
                   memlimit=5e8, # -- limit patches array to 500MB
                   ):
    def lab(msg):
        return '%s_fbncc_%s' % (prefix, msg)

    def get_rseed(name, N):
        fullname = lab(name)
        low = stable_hash(fullname) % (2 ** 31)
        rval = hp_choice(fullname, range(low, low + N))
        return rval

    patches = random_patches(
        Xcm, n_patches, size, size,
        rng=np_RandomState(get_rseed('patch_rseed', 10)),
        channel_major=True,
        memlimit=memlimit)

    remove_mean = hp_TF(lab('remove_mean'))
    beta = hp_lognormal(lab('beta'), np.log(100), np.log(100))
    hard_beta = hp_TF(lab('hard'))

    # TODO: use different nfilters, beta etc. for each algo

    # -- random projections filterbank allocation
    random_projections = partial(slm_fbncc_chmaj,
        m_fb=slm_uniform_M_FB(
            nfilters=n_filters,
            size=size,
            channels=pyll_getattr(Xcm, 'shape')[1],
            rseed=get_rseed('r_rseed', 10),
            normalize=hp_TF(lab('r_normalize')),
            dtype='float32',
            ret_cmajor=True,
            ),
        remove_mean=remove_mean,
        beta=beta,
        hard_beta=hard_beta)

    # -- random whitened projections filterbank allocation
    random_whitened_projections = partial(slm_fbncc_chmaj,
            m_fb=fb_whitened_projections(patches,
                patch_whitening_filterbank_X(patches,
                    gamma=hp_lognormal(lab('wr_gamma'),
                                       np.log(1e-2), np.log(100)),
                    o_ndim=2,
                    remove_mean=remove_mean,
                    beta=beta,
                    hard_beta=hard_beta,
                    ),
                n_filters=n_filters,
                rseed=get_rseed('wr_rseed', 10),
                dtype='float32',
                ),
            remove_mean=remove_mean,
            beta=beta,
            hard_beta=hard_beta)

    # -- whitened patches filterbank allocation
    whitened_patches = partial(slm_fbncc_chmaj,
            m_fb=fb_whitened_patches(patches,
                patch_whitening_filterbank_X(patches,
                    gamma=hp_lognormal(lab('wp_gamma'),
                                       np.log(1e-2), np.log(100)),
                    o_ndim=2,
                    remove_mean=remove_mean,
                    beta=beta,
                    hard_beta=hard_beta,
                    ),
                n_filters=n_filters,
                rseed=get_rseed('wp_rseed', 10),
                dtype='float32',
                ),
            remove_mean=remove_mean,
            beta=beta,
            hard_beta=hard_beta)

    # --> MORE FB LEARNING ALGOS HERE <--
    # TODO: V1-like filterbank (incl. with whitening matrix)
    # TODO: sparse coding
    # TODO: OMP from Coates 2011
    # TODO: K-means
    # TODO: RBM
    # TODO: DAA
    # TODO: ssRBM
    rchoice = hp_choice(lab('algo'), [
        random_projections,
        random_whitened_projections,
        whitened_patches,
        ])
    return rchoice


def pipeline_extension(prefix, X, n_patches, max_filters):
    assert max_filters > 16
    f_layer = new_fbncc_layer(prefix, X, n_patches,
            n_filters=s_int(
                hp_qloguniform('%sfb_nfilters' % prefix,
                    np.log(8.01), np.log(max_filters), q=16)),
            size=rfilter_size('%sfb_size' % prefix, 3, 8),
            )

    p_layer = partial(slm_lpool,
            stride=hp_choice('%sp_stride' % prefix, [1, 2]),
            order=hp_choice('%sp_order' % prefix,
                [1, 2, hp_lognormal('%sp_order_real' % prefix,
                    mu=np.log(1), sigma=np.log(3))]),
            ker_size=rfilter_size('%sp_size' % prefix, 2, 8))

    return [f_layer, p_layer]


def new_exit(pipeline, prefix):
    def lab(msg):
        return prefix % msg
    return {
        'pipe': pipeline,
        'remove_std0':
            hp_TF(lab('remove_std0')),
        'varthresh':
            hp_lognormal(lab('varthresh'),
                np.log(1e-4), np.log(1000)),
        'l2_reg': hp_lognormal(lab('l2_reg'),
            np.log(1e-5), np.log(1e3)),
        'divrowl2': hp_TF(lab('divrowl2')),
        }


def exit_grid(pipeline, layer_num, Xcm, n_patches, max_n_features):
    def lab(msg):
        return 'l%ieg_%s' % (layer_num, msg)

    fsize = rfilter_size(lab('fsize'), 3, 8)

    grid_res = hp_choice(lab('res'), [2, 3])
    grid_features_per_filter = 2 * (grid_res ** 2)
    grid_nfilters = max_n_features // grid_features_per_filter

    grid_filtering = new_fbncc_layer(
            prefix='l%ieg' % layer_num,
            Xcm=Xcm,
            n_patches=n_patches,
            n_filters=grid_nfilters,
            size=fsize,
            )

    grid_pooling = partial(slm_quantize_gridpool,
            alpha=hp_normal(lab('alpha'), 0.0, 1.0),
            use_mid=False,
            grid_res=grid_res,
            order=hp_choice(lab('order'), [
                1.0, 2.0, logu_range(lab('order_real'), .1, 10.)]))

    return new_exit(pipeline + [grid_filtering, grid_pooling], lab('%s'))


def exit_lpool_alpha(pipeline, layer_num, Xcm, n_patches, max_n_features):
    def lab(msg):
        return 'l%ielpa_%s' % (layer_num, msg)

    fsize = rfilter_size(lab('fsize'), 3, 8)
    filtering_res = pyll_getattr(Xcm, 'shape')[2] - fsize + 1
    # -- N.B. Xrows depends on other params, so we can't use it to set the
    #         upper bound on lpsize. We can only sample independently, and
    #         then fail below with non-positive number of features.
    size = rfilter_size(lab('lpsize'), 1, 5)
    stride = hp_choice(lab('stride'), [1, 2, 3])
    res = scope.ceildiv(scope.max(filtering_res - size + 1, 0), stride)
    if 0:
        # XXX: This is a smarter way to pick the n_filters, but it triggers
        # a bug in hyperopt.vectorize_helper.  The build_idxs_vals function
        # there needs to be smarter -- to recognize when wanted_idxs is a
        # necessarily subset of the all_idxs, and then not to append
        # wanted_idxs to the union defining all_idxs... because that creates a
        # cycle.  The trouble is specifically that lpool_res is used in the
        # switch statement below both in the condition and the response.
        nfilters = switch(res > 0,
            max_n_features // (2 * (res ** 2)),
            scope.Raise(ValueError, 'Non-positive number of features'))
    else:
        # this is less good because it risks dividing by zero,
        # and forces the bandit to catch weirder errors from new_fbncc_layer
        # caused by negative nfilters
        nfilters = max_n_features // (2 * (res ** 2))

    filtering = new_fbncc_layer(
            prefix='l%iel' % layer_num,
            Xcm=Xcm,
            n_patches=n_patches,
            n_filters=nfilters,
            size=fsize,
            )

    pooling = partial(slm_lpool_alpha,
            ker_size=size,
            stride=stride,
            alpha=hp_normal(lab('alpha'), 0.0, 1.0),
            order=hp_choice(lab('order_choice'), [
                1.0, 2.0, logu_range(lab('order_real'), .1, 10.)]))

    return new_exit(pipeline + [filtering, pooling], lab('%s'))


def exit_lpool(pipeline, layer_num, Xcm, n_patches, max_n_features):
    def lab(msg):
        return 'l%i_out_lp_%s' % (layer_num, msg)

    fsize = rfilter_size(lab('fsize'), 3, 8)
    filtering_res = pyll_getattr(Xcm, 'shape')[2] - fsize + 1
    # -- N.B. Xrows depends on other params, so we can't use it to set the
    #         upper bound on lpsize. We can only sample independently, and
    #         then fail below with non-positive number of features.
    psize = rfilter_size(lab('psize'), 1, 5)
    stride = hp_choice(lab('stride'), [1, 2, 3])
    pooling_res = scope.ceildiv(filtering_res - psize + 1, stride)
    nsize = rfilter_size(lab('nsize'), 1, 5)
    norm_res = pooling_res - nsize + 1

    # -- raises exception at rec_eval if norm_res is 0
    nfilters = max_n_features // (scope.max(norm_res, 0) ** 2)

    filtering = new_fbncc_layer(
            prefix='l%ielp' % layer_num,
            Xcm=Xcm,
            n_patches=n_patches,
            n_filters=nfilters,
            size=fsize,
            )

    pooling = partial(slm_lpool,
            ker_size=psize,
            stride=stride,
            order=hp_choice(lab('order_choice'), [
                1.0, 2.0, logu_range(lab('order_real'), .1, 10.)]))

    normalization = partial(slm_lnorm,
            ker_size=nsize,
            remove_mean=hp_TF(lab('norm_rmean')),
            threshold=hp_lognormal(lab('norm_thresh'),
                np.log(1.0), np.log(3)),
            )

    seq = hp_choice(lab('use_norm'), [
            [filtering, pooling],
            [filtering, pooling, normalization]])

    return new_exit(pipeline + seq, lab('%s'))


def pipeline_exits(pipeline, layer_num, Xcm, n_patches, max_n_features):
    grid = exit_grid(pipeline, layer_num, Xcm, n_patches, max_n_features)

    lpool_alpha = exit_lpool_alpha(pipeline, layer_num, Xcm, n_patches,
            max_n_features)

    lpool = exit_lpool(pipeline, layer_num, Xcm, n_patches, max_n_features)

    return [grid, lpool_alpha, lpool]


def uslm_domain(Xcm,
        batchsize,
        chmjr_image_shape,
        output_sizes,
        n_patches=50000,
        max_n_features=16000,
        max_layer_sizes=(64, 128),
        batched_lmap_speed_thresh=None,
        permit_affine_warp=True,
        abort_on_rows_larger_than=None,
        ):
    """
    This function works by creating a linear pipeline, with multiple exit
    points that could be the feature representation for classification.

    The function returns a switch among all of these exit points.
    """
    start_time = time.time()

    XC, XH, XW = chmjr_image_shape
    osize = hp_choice('warp_osize', output_sizes)

    assert XW > 3, chmjr_image_shape  # -- make sure we don't screw up channel-major

    warp_options = [
        # -- option 1: simple resize
        partial(slm_affine_image_warp,
            rot=0,
            shear=0,
            scale=[s_float(osize) / XH, s_float(osize) / XW],
            trans=[0, 0],
            oshape=[osize, osize]),
        ]
    if permit_affine_warp:
        # -- option 2: resize with rotation, shear, translation
        warp_options.append(
            partial(slm_affine_image_warp,
                rot=hp_uniform('warp_rot', low=-0.3, high=0.3),
                shear=hp_uniform('warp_shear', low=-0.3, high=0.3),
                # -- most of the scaling comes via osize
                scale=[
                    hp_uniform('warp_scale_h', low=0.8, high=1.2) * osize / XH,
                    hp_uniform('warp_scale_v', low=0.8, high=1.2) * osize / XW,
                    ],
                trans=[
                    hp_uniform('warp_trans_h', low=-0.2, high=0.2) * osize,
                    hp_uniform('warp_trans_v', low=-0.2, high=0.2) * osize,
                    ],
                oshape=[osize, osize]
                ))
    pipeline = [slm_img_uint8_to_float32,
                hp_choice('warp', warp_options)]
    Xcm = pyll_theano_batched_lmap(
        partial(callpipe1, pipeline),
        Xcm,
        batchsize=batchsize,
        print_progress_every=10,
        speed_thresh=batched_lmap_speed_thresh,
        abort_on_rows_larger_than=abort_on_rows_larger_than,
        x_dtype='uint8',
        )[:]

    exits = pipeline_exits(
                pipeline,
                layer_num=0,
                Xcm=Xcm,
                n_patches=n_patches,
                max_n_features=max_n_features)
    for layer_i, max_layer_size in enumerate(max_layer_sizes):
        extension = pipeline_extension(
                'l%i' % layer_i, Xcm, n_patches, max_layer_size)

        pipeline.extend(extension)
        Xcm = pyll_theano_batched_lmap(
                partial(callpipe1, extension),
                Xcm,  # scope.print_ndarray_summary('Xcm %i' % layer_i, Xcm),
                batchsize=batchsize,
                print_progress_every=10,
                speed_thresh=batched_lmap_speed_thresh,
                abort_on_rows_larger_than=abort_on_rows_larger_than,
                )[:]
        # -- indexing computes all the values (during rec_eval)
        exits.extend(
                pipeline_exits(
                    pipeline=pipeline,
                    layer_num=layer_i + 1,
                    Xcm=Xcm,
                    n_patches=n_patches,
                    max_n_features=max_n_features))

    return hp_choice("exit", exits)


class USLM_Exception(Exception):
    pass


def call_catching_pipeline_errors(fn):
    def raise_error(e):
        sio = StringIO.StringIO()
        traceback.print_exc(None, sio)
        tb = sio.getvalue()
        raise USLM_Exception(e, {
            'loss': float(1.0),
            'status': hyperopt.STATUS_FAIL,
            'failure': {
                'type': str(type(e)),
                'exc': repr(e),
                'tb': tb,
            }})
    try:
        return fn()
    except pyll_slm.InvalidDescription, e:
        raise_error(e)
    except pyll_slm.EvalTimeout, e:
        raise_error(e)
    except ZeroDivisionError, e:
        raise_error(e)
    except MemoryError, e:
        raise_error(e)
    except OSError, e:
        if 'allocate memory' in str(e):
            raise_error(e)
        else:
            raise
    except ValueError, e:
        if (('rowlen' in str(e) and 'exceeds limit' in str(e))
                or ('dimension mis-match' in str(e) and '= 0' in str(e))
                or ('had size 0' in str(e))
                or ('size on that axis is 0' in str(e))
                or ('non-finite features' in str(e))
                ):
            raise_error(e)
        else:
            raise
    except RuntimeError, e:
        if (('taking too long' in str(e))
                or ('allocate memory' in str(e))
                or ('kernel_reduce_sum' in str(e) and 'block: 0 x' in str(e))
                or ('CudaNdarray has dim 0' in str(e))
                ):
            raise_error(e)
        else:
            raise

