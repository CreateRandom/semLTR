import itertools
import multiprocessing
import os
import pickle
from functools import partial

import pandas as pd
import numpy as np

from extraction.corpus_constants_extraction import extract_idfs_for_all_unique_query_terms
from extraction.document_extraction import retrieve_docs_from_cache
from util.general_util import chunk_list


def extract_training_data(qrel_dict, pipeline, n_cores=-1, cache_path='doc_cache', verbose=True):
    """
    The main_logic entry point for generating training data from a query-relevance set
    :param qrels_file:  TREC format qrels file,
    :param queries_file: queries file (i.e. list of ids and query texts)
    :param pipeline: a pipeline in charge of extracting the features
    :return: LETOR format training data, frame with rel, qid, features
    """
    # query-document pairs
    qd_pairs = qrel_dict['qd_pairs']

    # all document ids this set refers to
    trec_ids = qrel_dict['trec_ids']
    trec_to_doc_dict = None
    if pipeline is not None:
    #    print('### Retrieving documents ###')
        # a dict that houses a Document for each id, read from the cache
   #     trec_to_doc_dict = retrieve_docs_from_cache(trec_ids, cache_path=cache_path,verbose=verbose,n_cores=n_cores)
        print('### Extracting features ###')
    # single process
    if n_cores is 1:
        total = extract_training_dataframe(qd_pairs,pipeline,cache_path=cache_path,verbose=verbose)
    # multiple processes
    else:
        # if the number is unspecified
        if n_cores is -1:
            # one process per core
            n_cores = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(processes=n_cores)

        # split into a list of lists, one per process
        chunked_key_lists = chunk_list(list(qd_pairs.keys()), n_cores)
        # chunked: each chunk is a subdictionary (i.e. a number of qd pairs)
        chunked = []

        # make a dict of lists, one list per core
        temp = [(i,[]) for i in range(n_cores)]
        doc_set_dict = dict(temp)

        for i, key_list in enumerate(chunked_key_lists):
            temp = {}
            for key in key_list:
                # get the qd_pair at this key
                pair = qd_pairs[key]
                # all the docs needed by this query
                docs = list(pair['rels'].keys())
                # look at the doc_set_dict, check whether there'd be overlap
                doc_set_dict[i].append(docs)
                temp[key] = qd_pairs[key]

            chunked.append(temp)

        doc_set_dict_flat = {}
        # flatten each of the sublists to get an overview of all the docs needed
        for core_key in doc_set_dict.keys():
            list_of_lists = doc_set_dict[core_key]
            doc_set_dict_flat[core_key] = list(itertools.chain(*list_of_lists))

        # generate all combinations of cores
        core_keys = list(doc_set_dict.keys())
        for pair in itertools.combinations(core_keys,2):
            # check if there is any overlap between the two
            docs_0 = doc_set_dict_flat[pair[0]]
            docs_1 = doc_set_dict_flat[pair[1]]
            print(pair)
            overlap = [id for id in docs_0 if id in docs_1]
            print(len(overlap))


        # use the partial functool here: this allows us to keep some paramters fixed
        # the multiprocessing logic will thus only operate on the first parameter,
        # passing different chunks to the extraction function

        # we get a list of dataframes that house X, y, ids (one for each process)
        pool_outputs = pool.map(partial(extract_training_dataframe, pipeline=pipeline,
                                        cache_path=cache_path, verbose=verbose),
                                chunked)

        # concat all the frames into a single one
        total = pd.concat(pool_outputs)

        # get rid of the pool
        pool.close()
        pool.join()

    return total


def extract_training_dataframe(qd_pairs, pipeline,cache_path,verbose=False):
    """

    :param qd_pairs: query-document pairs
    :param trec_to_doc_dict: a dict that houses a document for each TREC id
    :param pipeline: the pipeline to use to extract features
    :return: X, y and ids for use in LETOR learning
    """

    # in case the multithreading logic passes this function an empty task
    # just return an empty frame
    if(len(qd_pairs) == 0):
        return pd.DataFrame()
    idf_values, mean_body_length, mean_title_length = None, None, None
    if pipeline is not None:
        idf_values, mean_body_length, mean_title_length = extract_idfs_for_all_unique_query_terms(qd_pairs)

    # Loop over all queries in our set to extract X and y vectors
    # based on query text / features and relevance ground truth
    all_frames = []
    for query_id in qd_pairs:
        query = qd_pairs[query_id]
        # dict from trec_ids to scores
        rel_docs = query['rels']
        # if there's a pipeline to run
        if pipeline is not None:
            print('Extracting features for ' + query['text'])
            # get the DataFrame for all docs relative to this query
            all_doc_ids = list(rel_docs.keys())
            # load the relevant documents from the disk
            trec_to_doc_dict = retrieve_docs_from_cache(list_trec_ids=all_doc_ids,cache_path=cache_path,verbose=verbose,n_cores=1)

            documents = [trec_to_doc_dict[id] for id in rel_docs.keys()]
            frame = pipeline.extract_all_features(query['text'], documents,**{'idf_values': idf_values,
                                                                              'mean_body_length': mean_body_length,
                                                                              'mean_title_length': mean_title_length})
        # if not, only return ys and ids
        else:
            frame = pd.DataFrame()
        # relevance values
        y = rel_docs.values()
        # query ids --> same for each doc
        ids = [query_id] * len(rel_docs)
        # sort frame alphabetically
        frame = frame.reindex(sorted(frame.columns), axis=1)

        # add y and ids to the feature frame
        frame['y'] = y
        frame['ids'] = ids

        all_frames.append(frame)
    # concatenating the frames for all queries processed
    return pd.concat(all_frames)


def extract_letor_arrays(frame):
    """
        Extracts X, y and ids arrays from data frame
    :param frame: A dataframe generated by our extraction code
    """
    # the last two columns are y and ids, so don't include them
    if len(frame.columns) > 2:
        X = frame[frame.columns[:-2]].values
    else:
        X = None
    y = frame['y'].values
    ids = frame['ids'].values
    return X, y, ids


def write_model_to_cache(model_name,model,cache_path):
    write_features_to_cache(model_name + '_model',model,[],cache_path)

def retrieve_model_from_cache(model_name,cache_path):
    model, _ = retrieve_features_from_cache(model_name,cache_path)
    return model

def write_features_to_cache(name, feature_array, feature_name_list, cache_path):
    index_dict = {}
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    # check whether we have a pickled index
    index_path = os.path.join(cache_path, 'index.p')
    if (os.path.exists(index_path)):
        # if so, load it up
        with open(index_path, 'rb') as pfile:
            # load up the dict
            index_dict = pickle.load(pfile)

    # write the array to a file
    file_name = name + '.f'
    file_path = os.path.join(cache_path, file_name)
    with open(file_path, 'wb') as pfile:
        array_and_name = (feature_array, feature_name_list)
        pickle.dump(array_and_name, pfile)

    index_dict[name] = file_name

    # dump the index
    with open(index_path, 'wb') as pfile:
        pickle.dump(index_dict, pfile)


def retrieve_features_from_cache(name, cache_path):
    """
    Retrieves Document for each trec id in the list, relying on a cache specified in the path
    if the cache doesn't exist yet, it'll be created and content will be retrieved from the web
    and added.
    :param list_trec_ids: List of TREC ids for each of which a document is retrieved
    :param cache_path: The path to cache to
    :return: A dictionary from ID to document objects
    """
    index_dict = {}

    index_path = os.path.join(cache_path, 'index.p')
    if (os.path.exists(index_path)):
        # if so, load it up
        with open(index_path, 'rb') as pfile:
            # load up the dict
            index_dict = pickle.load(pfile)

    feature_array = None
    feature_names = None
    if name in index_dict:
        file_name = index_dict[name]
        file_path = os.path.join(cache_path, file_name)

        with open(file_path, 'rb') as pfile:
            # load up the dict
            array_and_name = pickle.load(pfile)
            feature_array, feature_names = array_and_name

    return feature_array, feature_names