import operator

import numpy as np
import pyltr

from chatnoir import chatnoir_api as api
from extraction.corpus_constants_extraction import extract_idfs_for_all_unique_query_terms
from extraction.document_extraction import retrieve_docs, retrieve_docs_from_cache


def generate_pred_vector_from_ranks(list_rel_docs, k_highest, scores=None):
    """

    :param list_rel_docs: list of docs referred to in the ground truth
    :param k_highest: a list of retrieved ids, sorted by rank
    :param scores: scores to use for re-ranking elements in k_highest
    :return: a pred vector used by pyltr for computing metrics, it has
    a score for each document in the ground truth and is then used
    to rank the documents and to compute NDCG
    """
    doc_scores = {}
    for doc_id in list_rel_docs:
        # this document wasn't retrieved, assign zero
        if doc_id not in k_highest:
            score = 0
        else:
            if scores is None:
                # this document was retrieved, assign 1 / rank
                score = 1 / (k_highest.index(doc_id) + 1)
            # we have a score (e.g. computed by a model), assign that instead
            else:
                score = scores[doc_id]

        doc_scores[doc_id] = score

    pred = np.zeros(len(doc_scores.keys()))
    index = 0
    for id in doc_scores.keys():
        pred[index] = doc_scores[id]
        index = index + 1

    return pred, doc_scores


def perform_ranking(qrel_dict, k, pipeline=None, model=None, metric=None,
                    cache_path='doc_cache', qids=None, verbose=None):
    """
    Ranks the queries in qrel_dict (or optionally only those in qids). Default
    behaviour: Just returns score of k best docs ranked by ChatNoir. If a pipeline
    and a model are provided, also perform re-ranking by extracting features using the
    pipeline and

    :param qrel_dict: qrels
    :param k: how many documents to retrieve initially
    :param pipeline: a pipeline for feature extraction
    :param model: a model for re-ranking
    :param metric: the metric to compute
    :param cache_path: a path to the doc cache
    :param qids: qids for which to perform (re-)ranking, if None: (re-)rank all
    :param verbose: print out resulting rankings
    :return:
    """
    qd_pairs = qrel_dict['qd_pairs']
    if qids is None:
        qids = qd_pairs.keys()
    else:
        # make sure every id only appears once
        qids = set(qids)

    if metric is None:
        metric = pyltr.metrics.NDCG(k=20)

    doc_by_doc_judgments = {}
    all_ndcgs = {}

    # query id --> list of docs judged relevant by ChatNoir
    chatnoir_docs = {}
    # all ids we'll need to retrieve in this go
    all_ids = []
    for query_id in qids:
        query = qd_pairs[query_id]
        ranked_ids = api.get_chatnoir_ranking(query['text'], k)
        chatnoir_docs[query_id] = ranked_ids
        all_ids.extend(ranked_ids)

    trec_to_doc_dict = None
    if pipeline is not None:
        # grab all docs
        trec_to_doc_dict = retrieve_docs_from_cache(all_ids, cache_path)

    # Loop over all queries in the set
    for query_id in chatnoir_docs.keys():
        query = qd_pairs[query_id]
        print('Re-Ranking query: ' + query['text'])
        ranked_ids = chatnoir_docs[query_id]

        score_dict = None
        if pipeline is not None and model is not None:
            if verbose: print('Re-ranking')
            documents = [trec_to_doc_dict[id] for id in ranked_ids]
            idf_values, mean_body_length, mean_title_length = extract_idfs_for_all_unique_query_terms(qd_pairs)
            x = pipeline.extract_all_features(query['text'], documents,**{'idf_values': idf_values,
                                                                              'mean_body_length': mean_body_length,
                                                                              'mean_title_length': mean_title_length}).values

            # predictions for all the documents returned by ChatNoir
            pred = model.predict(x)
            # stuff ids and scores into a dictionary
            # id --> score
            score_dict = dict(zip(ranked_ids, pred))

        # trec_ids in ground truth that have a relevance ranking
        rel_docs = query['rels'].keys()

        # zip up the relevance judgements with the ranked ids
        # for each of the ground truth documents, assign the score of the model
        # and zero otherwise
        pred, doc_judgments = generate_pred_vector_from_ranks(rel_docs, ranked_ids, scores=score_dict)
        y = np.array(list(query['rels'].values()))
        ids = np.array([int(query_id) for _ in range(len(rel_docs))])
        # ndcg for this query
        ndcg = metric.calc_mean(ids, y, pred)
        all_ndcgs = {**all_ndcgs, **{query_id : ndcg}}
        doc_by_doc_judgments = {**doc_by_doc_judgments, **{query['qid']: doc_judgments}}

        if verbose:
            print('NDCG = ' + str(ndcg))
            print_doc_judgments(qrel_dict, {query_id : doc_judgments})

    # return per-query ndcgs
    return all_ndcgs, doc_by_doc_judgments


def print_doc_judgments(qrels, judgment_dict):
    for qid in judgment_dict.keys():
        doc_scores = judgment_dict[qid]
        query = qrels['qd_pairs'][qid]
        tuples = []

        for trec_id in doc_scores:

            rel = query['rels'][trec_id]
            score = doc_scores[trec_id]
            if score > 0:
                tuples.append((score,rel,trec_id))
        tuples.sort(key=operator.itemgetter(0),reverse=True)

        for t in tuples:
            # sort by score
            doc = retrieve_docs([trec_id])[trec_id]
            title = str(doc.field_dict['title']['text'])
            score, rel, trec_id = t
            print('Rank = ' + str(1 / score) + '. Rel: ' + str(rel) + '. Title:' + title)