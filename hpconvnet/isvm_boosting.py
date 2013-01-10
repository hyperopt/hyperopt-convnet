"""
Experiment classes
"""

import copy
import logging

import numpy as np

import hyperopt
from hyperopt.base import trials_from_docs

logger = logging.getLogger(__name__)

class BoostHelper(object):
    """
    Several helper-methods related to boosting that work on the basis of
    a dictionary doc_by_tid rather than the original trials object.

    _continuing: map `tid` -> list of documents that continue `tid`.
        Forward arrows in the exploration of ensembles.

    """

    @staticmethod
    def query_MongoTrials(mt, fields=(), query=None):
        """
        Retrieve `docs` for the constructor without downloading the
        entire experiment history, as would be done by mt.refresh().
        """
        exp_key = mt._exp_key
        if query is None:
            query = {}
        else:
            query = copy.deepcopy(dict(query))
        if exp_key != None:
            query.setdefault('exp_key', exp_key)
        query.setdefault('state', {'$ne': hyperopt.JOB_STATE_ERROR})
        rval = mt.handle.jobs.find(
            query,
            fields=[
                'tid',
                'result.status',
                'result.loss',
                'misc.boosting.continues',
                'misc.boosting.variant', # -- optional
                '_attachments',
            ] + list(fields),
            snapshot=True)
        if rval is None:
            return []
        else:
            return list(rval)

    def __init__(self, docs, doc_by_tid=None):
        if doc_by_tid is None:
            self.doc_by_tid = dict([(d['tid'], d) for d in docs])
            # -- assert that every document has a unique tid
            assert len(self.doc_by_tid) == len(docs)
        else:
            self.doc_by_tid = doc_by_tid

        self._continuing = {}
        for d in docs:
            pre_tid = self.continues(d)
            if pre_tid is not None:
                pre_tid = pre_tid['tid']
            self._continuing.setdefault(pre_tid, []).append(d)


    def ok_tids(self):
        return [d['tid'] for d in self.doc_by_tid.values()
                if d['result']['status'] == hyperopt.STATUS_OK]

    def best_doc(self):
        # -- it is no longer the case that losses can be compared
        #    between generations, only between siblings is ok.
        raise NotImplementedError('since hacking bagging support')
        ok_tids = self.ok_tids()
        losses = [self.doc_by_tid[tid]['result']['loss'] for tid in ok_tids]
        assert None not in losses
        best_idx = np.argmin(losses)
        rval = self.doc_by_tid[ok_tids[best_idx]]
        return rval

    def continues(self, doc):
        """Returns the (older-than-doc) trial whose decisions `doc` built on.
        """
        # -- hack to support `doc` that is a misc sub-doc
        doc = self.doc_by_tid[doc['tid']]
        if 'boosting' in doc['misc']:
            rval_tid = doc['misc']['boosting']['continues']
            if rval_tid is None:
                return None
            else:
                return self.doc_by_tid[rval_tid]
        else:
            return None

    def continuing(self, doc):
        """Returns all docs whose decisions were built on `doc`.
        """
        if doc is None:
            return self._continuing.get(None, [])
        else:
            return self._continuing.get(doc['tid'], [])

    def children(self, doc):
        return self.continuing(doc)

    def best_child(self, doc):
        losses = [(child['result']['loss'], child)
                  for child in self.children(doc)
                  if child['result']['status'] == hyperopt.STATUS_OK
                 ]
        if losses:
            losses.sort()
            return losses[0][1]
        else:
            return None

    def ensemble_members(self):
        """Return all docs that are part of the best ensemble in order of
        addition to the ensemble.
        """
        # function works by working backward through the
        # doc['misc']['boosting']['continues'] links
        reversed_members = [self.best_doc()]
        while self.continues(reversed_members[-1]) != None:
            reversed_members.append(self.continues(reversed_members[-1]))
        rval = list(reversed(reversed_members))
        return rval

    def history(self, doc):
        """
        Return a list of documents that continued previous ones, leading
        eventually to `doc`.  `doc` itself is the last element.
        """
        rval = []
        try:
            doc_or_None = self.continues(doc)
            while doc_or_None is not None:
                rval.append(doc_or_None)
                doc_or_None = self.continues(doc_or_None)
            rval.reverse()
            rval.append(doc)
            return rval
        except KeyError, e:
            if 'boosting'in str(e):
                return [doc]
            raise


def suggest(
    new_ids, domain, trials, sub_suggest,
    min_ok_per_round=1,
    min_valid_per_round=1,
    absolute_loss_thresh=1.0,
    relative_loss_thresh=None,
    ):
    """

    Parameters
    ----------

    min_ok_per_round - int
        A trial cannot be extended in the ensemble until it has this many
        siblings with status 'ok' and a loss <= absolute_loss_thresh.

    min_valid_per_round - int
        A trial cannot be extended in the ensemble until it has this many
        siblings whose job state is not ERROR.

    absolute_loss_thresh - float
        Jobs with loss greater than this are not counted as 'ok'.

    relative_loss_thresh - None or float
        A child cannot become a parent in the ensemble unless it improves on its
        parent with a loss <= relative_loss_thresh * parent_loss.

    This search algo works by injecting a ['misc']['boosting'] subdocument into
    every trial, with keys:
      * variant - identify the type of boosting at work
      * continues - the trial ID (tid) of the previously selected trial in the
                    ensemble, or `None` for first-round trials

    In order for boosting to work properly, the 'loss' reported by trial must
    represent the CUMULATIVE ENSEMBLE LOSS if the ensemble were to be extended
    to include that particular trial.

    """
    new_id, = new_ids

    valid_docs = [t for t in trials
            if t['state'] != hyperopt.JOB_STATE_ERROR]

    # -- ok_docs are those which are eligible to be a member of the
    #    final ensemble.
    ok_docs = [t for t in valid_docs
            if t['result']['status'] == hyperopt.STATUS_OK
               and t['result']['loss'] <= absolute_loss_thresh]

    logger.info('n_ok: %i n_valid: %i' % (len(ok_docs), len(valid_docs)))

    valid_helper = BoostHelper(valid_docs)
    ok_helper = BoostHelper(ok_docs)

    cur_parent = None
    cur_parent_tid = None
    while True:
        n_ok_children = len(ok_helper.children(cur_parent))
        n_valid_children = len(valid_helper.children(cur_parent))
        logger.info('cur_parent: %s  n_ok_children: %i  n_valid_children: %i'
                    % (None if cur_parent is None else cur_parent['tid'],
                       n_ok_children,
                       n_valid_children))
        if n_ok_children < min_ok_per_round:
            break
        if n_valid_children < min_valid_per_round:
            break

        best_child = ok_helper.best_child(cur_parent)
        assert best_child is not None  # -- because ok_helper has some elements

        if None not in (cur_parent, relative_loss_thresh):
            rel_thresh = cur_parent['result']['loss'] * relative_loss_thresh
            if best_child['result']['loss'] >= rel_thresh:
                break

        logger.info('best_child: %i' % best_child['tid'])
        cur_parent = best_child
        cur_parent_tid = best_child['tid']
        del best_child

    cur_siblings = valid_helper.children(cur_parent)

    current_trials = trials_from_docs(
            cur_siblings,
            exp_key=trials._exp_key,
            # -- validate=False is much faster
            validate=False)

    new_trial_docs = sub_suggest([new_id], domain, current_trials)

    for trial in new_trial_docs:
        misc = trial['misc']
        # -- boosting cannot be nested with current data structure
        assert 'boosting' not in misc
        # -- I think the following was a debugging sanity check
        assert trial['tid'] == new_id
        misc['boosting'] = {
            'variant': {
                'name': 'async_suggest',
                'min_ok_per_round': min_ok_per_round,
                'min_valid_per_round': min_valid_per_round,
                'relative_loss_thresh': relative_loss_thresh,
                'absolute_loss_thresh': absolute_loss_thresh,
                },
            'continues': cur_parent_tid}

    return new_trial_docs

