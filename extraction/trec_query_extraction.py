import random
from collections import OrderedDict

import numpy as np


def load_qrels(qrels_file, queries_file):
    """

    :param qrels_file: file path to qrels file in TREC format
    :param queries_file file path to queries in TREC format
    :return: list of queries, each query is a dict with id, text and rels (dict with pairs and info)
    """
    qd_pairs = {}
    n_query_document_pairs = 0
    n_filtered = 0
    # all trec ids in this query-document collection
    trec_ids = []

    query_texts = {}

    # first, load up the query file
    # XML format
    if queries_file.endswith('.xml'):
        query_texts = load_query_xml(queries_file)
    # text format
    else:
        with open(queries_file) as queries:
            # extract the texts for all query texts
            for line in queries:
                (key, val) = line.split(sep=':')
                query_texts[key] = str.rstrip(val)
    with open(qrels_file) as qrels:

        # extract all query-document pairs
        for line in qrels:
            # 101 0 clueweb09-en0019-32-21921 2
            (qid, _, trec_id, rel) = line.split()
            # filter out spam documents --> rel = -2
            if int(rel) >= 0:
                trec_ids.append(trec_id)
                # if we haven't seen this query yet
                if qid not in qd_pairs:
                    rels = OrderedDict()
                    rels[trec_id] = int(rel)

                    qd_pairs[qid] = {'qid': qid, 'text': query_texts[qid], 'rels': rels}
                # if we have, just add the relevance score to the dict
                else:
                    qd_pairs[qid]['rels'][trec_id] = int(rel)
                n_query_document_pairs += 1
            else:
                n_filtered += 1

    to_return = {}
    print('Filtered ' + str(n_filtered) + ' spam documents.')
    to_return['n_pairs'] = n_query_document_pairs
    to_return['qd_pairs'] = qd_pairs
    to_return['trec_ids'] = trec_ids
    return to_return


def load_query_xml(xml_path):
    """
    Helper for loading up query file in xml format
    :param xml_path: a path
    :return: dict from id to query string
    """
    from xml.etree import cElementTree as ET

    with open(xml_path, 'r') as myfile:
        data = myfile.read()

    root = ET.fromstring(data)

    queries = {}

    for topic in list(root):
        qid = topic.attrib['number']
        query_text = topic.find('query').text
        if query_text is not None:
            queries[qid] = query_text

    return queries

def find_all_ids(qd_pairs):
    """
    A helper to find a list of all ids in a query set
    :param qd_pairs: a query set
    :return: the list of ids and the number of pairs
    """
    all_trec_ids = []
    n_pairs = 0
    for query_id in qd_pairs.keys():
        query = qd_pairs[query_id]
        # dict from trec_ids to scores
        rel_docs = query['rels']
        trec_ids = rel_docs.keys()
        all_trec_ids.extend(trec_ids)
        n_pairs = n_pairs + len(trec_ids)

    return all_trec_ids, n_pairs


def enforce_order_on_dict(dict, order):
    tuples = [(key, dict[key]) for key in order]
    return OrderedDict(tuples)

def merge_qrels(list_qrel_dicts):
    """
    For merging a list of qrel dicts into a single qrel dict
    :param list_qrel_dicts: a list of qrels
    :return: a single qrel
    """
    all_qrels = {}
    for qrel_dict in list_qrel_dicts:
        for key in qrel_dict.keys():
            if key in all_qrels:
                if isinstance(qrel_dict[key], dict):
                    all_qrels[key] = {**all_qrels[key], **qrel_dict[key]}
                if isinstance(qrel_dict[key], list):
                    all_qrels[key].extend(qrel_dict[key])
                if isinstance(qrel_dict[key], int):
                    all_qrels[key] = all_qrels[key] + qrel_dict[key]
            else:
                all_qrels[key] = qrel_dict[key]
    # shuffle the qd pairs
    pairs = all_qrels['qd_pairs']
    ids = list(pairs.keys())
    random.shuffle(ids)
    shuffled_pairs = {}
    for id in ids:
        shuffled_pairs = {**shuffled_pairs, **{id : pairs[id]}}
    all_qrels['qd_pairs'] = shuffled_pairs

    return all_qrels


def split_qrels(qrel_dict, validation_size=0.2):
    """
    A helper function for splitting a set of queries into a training and a
    validation set.
    :param qrel_dict: the total set of queries
    :param validation_size: the percentage of queries to form the validation set
    :return:
    """
    # the queries with associated relevance info --> split these up
    qd_pairs = qrel_dict['qd_pairs']

    query_ids = qd_pairs.keys()
    # the number of queries
    n_queries = len(query_ids)
    # the size of the validation set
    n_validation = int(np.floor(float(n_queries) * validation_size))
    # keep the seed fixed to have reproducible outcomes
    np.random.seed(42)
    # sample the ids
    validation_ids = np.random.choice(list(query_ids), size=n_validation, replace=False)

    # split the qd_pairs
    train_qd = {x: qd_pairs[x] for x in qd_pairs if x not in validation_ids}
    validation_qd = {x: qd_pairs[x] for x in qd_pairs if x in validation_ids}

    # now also filter the trec_ids and update the n_pairs
    train_ids, n_t_ids = find_all_ids(train_qd)
    validation_ids, n_v_ids = find_all_ids(validation_qd)

    # re-compile the three bits into a new dict
    qrel_train = {'qd_pairs': train_qd, 'n_pairs': n_t_ids, 'trec_ids': train_ids}
    qrel_validation = {'qd_pairs': validation_qd, 'n_pairs': n_v_ids, 'trec_ids': validation_ids}
    return qrel_train, qrel_validation
