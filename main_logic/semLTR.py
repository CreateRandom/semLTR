import datetime
import itertools
import os
import pickle
from collections import OrderedDict

from functools import partial

import pyltr
import multiprocessing

from gensim.models import KeyedVectors
from pyltr.util.group import get_groups

from evaluation.baseline_evaluation import perform_ranking
from extraction.training_extraction import extract_training_data, extract_letor_arrays, write_features_to_cache, \
    retrieve_features_from_cache, write_model_to_cache
from extraction.trec_query_extraction import load_qrels, merge_qrels
from util.general_util import chunk_list

from extraction.feature_extraction import FeaturePipeline, DocumentLengthExtractor, LMIRExtractor, W2VExtractor, BM25Extractor, SubjObjRelationExtractor
from extraction.document_extraction import retrieve_docs_from_cache
from sklearn.model_selection import GroupKFold, ParameterGrid
import numpy as np


def load_and_merge_trec_qrels():
    ### DATA LOADING ###

    qrels_2010 = load_qrels('../trec_data/trec_2010/qrels.adhoc', '../trec_data/trec_2010/wt2010-topics.xml')
    qrels_2011 = load_qrels('../trec_data/trec_2011/qrels.adhoc', '../trec_data/trec_2011/queries.txt')
    qrels_2012 = load_qrels('../trec_data/trec_2012/qrels.adhoc', '../trec_data/trec_2012/queries.txt')

    ### MERGE INTO ONE SET ###
    qrels_all = merge_qrels([qrels_2010, qrels_2011, qrels_2012])
    return qrels_all


def perform_experiments(qrels, extract_docs=True, perform_CV=True):

    # extract documents batchwise beforehand
    if extract_docs:
        extract_docs_for_rels(qrels)

    ### SETTING UP CONDITIONS

    # the limit parameter controls how many word vectors will be loaded from the disk
    # GoogleNews has 3 million words, this requires 3.5 GB of RAM
    # We can reduce memory consumption by 5/6 by only loading the 500.000 most frequent words
    model = KeyedVectors.load_word2vec_format('../pretrained/GoogleNews-vectors-negative300.bin', binary=True,
                                              limit=500000)

    # disable the most_similar operation to save more memory, see here https://stackoverflow.com/a/50496705
    model.init_sims(replace=True)

    extractor_sets = {'letor': [DocumentLengthExtractor(), BM25Extractor(), LMIRExtractor()],
                      'w2v': [W2VExtractor(w2v_model=model)],
                      'sub_obj': [SubjObjRelationExtractor()],
                      }

    # conditions to compare
    conditions = OrderedDict([
        ('no_reranking', None),
        ('letor', FeaturePipeline(extractor_sets['letor'])),
        ('w2v', FeaturePipeline(extractor_sets['w2v'])),
        ('sub_obj', FeaturePipeline(extractor_sets['sub_obj'])),
        ('all', ['letor', 'w2v', 'sub_obj'])
    ])

    cache_path = 'feature_cache'


    if perform_CV:

        # We chose the metric to be normalized discounted gain with a cutoff of 20
        metric = pyltr.metrics.NDCG(k=20)

        results = run_CV_for_conditions(qrels=qrels, conditions=conditions, metric=metric, cache_path=cache_path,
                                        n_docs=100)

        # write result dict to path
        file_path = os.path.join(cache_path, 'results.dict')
        with open(file_path, 'wb') as pfile:
            pickle.dump(results, pfile)

def run_CV_for_conditions(conditions, qrels, metric, n_folds=5, n_docs = 100,
                          cache_path=None,n_cores_extraction=1, n_cores_grid_search=8):

    # the path to cache this run to
    if cache_path is None:
        cache_path = 'run_' + datetime.datetime.now().ctime().replace(" ", "_")

    # empty pipeline, only extract y and ids and use those throughout
    training_frame = extract_training_data(qrels, pipeline=None, n_cores=1, verbose=True)

    _, y, ids = extract_letor_arrays(training_frame)

    # grouped cross-validation
    # use query_ids as groups here
    # to make sure queries aren't split across folds
    group_kfold = GroupKFold(n_splits=n_folds)

    fold_indices = []

    # generate folds and use them for all conditions
    for train_inner_index, test_inner_index in group_kfold.split(y, groups=ids):
        fold_indices.append((train_inner_index, test_inner_index))


    # store per-query NDCGs as well as fold means
    condition_results = {}

    for condition_name in conditions.keys():
        print('########################################################')
        print('Scoring condition ' + condition_name)

        ndcgs_pure_fold_means = []
        ndcgs_reranked_fold_means = []
        all_scores_ranked = []
        all_doc_by_doc_judgments = []
        feature_importances = []

        # best parameters for every fold
        best_configs = []

        X = None
        pipeline_to_use = None

        # first, find the pipeline
        conditions_pipeline_s = conditions[condition_name]
        # if this is a combination of pipelines to be run
        # merge them into a single one
        if isinstance(conditions_pipeline_s,list):
            pipelines = [conditions[pipeline_name] for pipeline_name in conditions_pipeline_s]
            # merge into single pipeline
            all_extractors = []
            for pipeline in pipelines:
                all_extractors.extend(pipeline.extractors)
            pipeline_to_use = FeaturePipeline(all_extractors)

            feature_arrays = []
            feature_names = []
            # get the dataframes already extracted by the pipelines in the list
            for pipeline_name in conditions_pipeline_s:
                # load the features from the disk
                array, names = retrieve_features_from_cache(pipeline_name, cache_path)
                if array is not None and feature_names is not None:
                    feature_arrays.append(array)
                    feature_names.append(names)


            # merge into single frame
            X = np.concatenate(feature_arrays,axis=1)
            # flatten names into list
            feature_names = list(itertools.chain(*feature_names))

        # this is a single pipeline object
        else:
            pipeline_to_use = conditions_pipeline_s

            # try to get the stuff from the cache
            X, feature_names = retrieve_features_from_cache(condition_name, cache_path)
            if X is None:
                # generate the training data and store it
                frame = extract_training_data(qrels, pipeline_to_use, n_cores=n_cores_extraction)
                X, _, _ = extract_letor_arrays(frame)
                # feature names

                feature_names = frame.columns[:-2].tolist()
                # store the features on the disk
                write_features_to_cache(condition_name, X, feature_names, cache_path)

        fold_counter = 1

        # the actual cross-validation logic
        for train_inner_index, test_inner_index in fold_indices:
            print('Fold ' + str(fold_counter))
            fold_counter = fold_counter + 1

            # the query ids
            ids_train, ids_test = ids[train_inner_index], ids[test_inner_index]

            # the relevance judgments
            y_train, y_test = y[train_inner_index], y[test_inner_index]

            model = None
            # train the model on this fold if we have features
            if X is not None:

                X_train, X_test = X[train_inner_index], X[test_inner_index]

                perform_inner = True

                if perform_inner:
                    # parameter optimization
                    # pick a number between 100 and 1000
                    n_trees = 100

                    grid_dict = OrderedDict([
                        ('n_trees', [n_trees]),
                        ('max_depth', [4, 8]),
                        ('learning_rate', [5 / n_trees, 10 / n_trees]),
                        ('max_features', ['sqrt', None])
                    ])

                    grid = ParameterGrid(grid_dict)

                    # inner / nested cross-validation
                    nested_k_fold = GroupKFold(n_splits=2)

                    pool = multiprocessing.Pool(processes=n_cores_grid_search)

                    # split into a list of lists, one per process
                    chunked_grid = chunk_list(list(grid), n_cores_grid_search)

                    # iterator over inner CV splits
                    split_iterator = list(nested_k_fold.split(X=X_train, y=y_train, groups=ids_train))

                    cv_func = partial(run_and_evaluate_configs,metric=metric,X_train=X_train,
                                      y_train=y_train,ids_train =ids_train, split_iterator = split_iterator)

                    # we get a list of lists of tuples (config_dict, mean_score)
                    pool_outputs = pool.map(cv_func,chunked_grid)

                    # flatten everything
                    config_scores = list(itertools.chain(*pool_outputs))

                    # get rid of the pool
                    pool.close()
                    pool.join()

                    max_score = 0
                    best_config = {}
                    for tuple in config_scores:
                        config_dict, score = tuple
                        if score > max_score:
                            max_score = score
                            best_config = config_dict

                    best_configs.append(best_config)

                    print(config_scores)

                    print('Best config :' + str(best_config))

                else:
                    best_config = {'n_trees' : 100, 'learning_rate' : 0.05,
                                   'max_depth' : 4, 'max_features' : 'sqrt'}

                # train on the full set
                n_trees = best_config['n_trees']
                learning_rate = best_config['learning_rate']
                max_depth = best_config['max_depth']
                max_features = best_config['max_features']

                # train the model with these parameters
                model = pyltr.models.LambdaMART(
                    metric=metric,
                    n_estimators=n_trees,
                    learning_rate=learning_rate,
                    max_features=max_features,
                    query_subsample=0.5,
                    max_depth=max_depth,
                    verbose=1,
                    # to make sure previous solution is erased after subsequent call to fit
                    warm_start=False
                )

                model.fit(X_train,y_train,ids_train)

                # store the model
                write_model_to_cache(condition_name,model,cache_path)

                # get the feature importance scores
                imps = model.feature_importances_
                # store them for now
                feature_importances.append(imps)

                # finally, get the pure model NDCG
                # and make predictions on the test set
                pred = model.predict(X_test)
                ndcgs_model = metric.calc_mean(ids_test, y_test, pred)
                ndcgs_pure_fold_means.append(ndcgs_model)
                print('Pure model :' + str(ndcgs_model))


            # if there's no model for re-ranking, retrieve exactly as many documents
            # as will be looked at by the ranking
            if model is None:
                n_docs = metric.k

            # rank for this condition
            all_query_ndcgs, doc_by_doc_judgments = perform_ranking(qrels, k=n_docs, pipeline=pipeline_to_use, model=model, qids=ids_test,
                                              metric=metric)

            print('Re-ranked : ' + str(all_query_ndcgs))

            # store the scores
            all_scores_ranked.append(all_query_ndcgs)

            all_doc_by_doc_judgments.append(doc_by_doc_judgments)

            # and the mean over this fold
            ndcgs_reranked_fold_means.append(np.mean(list(all_query_ndcgs.values())))

        name_to_importance_dict = {}
        if feature_importances:
            # compute the mean of feature importances over folds
            feature_importances = np.array(feature_importances)
            # column wise means, so axis = 0
            mean_imps = np.mean(feature_importances, axis=0)
            # zip them up with the names to have any hang of what the values mean
            name_to_importance_dict = dict(zip(feature_names,mean_imps))

        condition_results[condition_name] = {'all_scores_ranked': all_scores_ranked,
                                             'doc_by_doc_judgments': all_doc_by_doc_judgments,
                                             'fold_means_model' : ndcgs_pure_fold_means,
                                             'fold_means_reranked': ndcgs_reranked_fold_means,
                                             'best_configs_per_fold': best_configs,
                                             'fold_means_featureImportance' : name_to_importance_dict}



    return condition_results


def run_and_evaluate_configs(grid, metric, X_train, y_train, ids_train, split_iterator):

    config_scores = {}

    for i, config in enumerate(grid):
        for train_inner_index, test_inner_index in split_iterator:

            X_inner_train = X_train[train_inner_index]
            y_inner_train = y_train[train_inner_index]
            ids_inner_train = ids_train[train_inner_index]

            n_trees = config['n_trees']
            learning_rate = config['learning_rate']
            max_depth = config['max_depth']
            max_features = config['max_features']

            # train the model with these parameters
            model = pyltr.models.LambdaMART(
                metric=metric,
                n_estimators=n_trees,
                learning_rate=learning_rate,
                max_features=max_features,
                query_subsample=0.5,
                max_depth=max_depth,
                verbose=1,
                # to make sure previous solution is erased after subsequent call to fit
                warm_start=False
            )
            print('Training with ' + str(config))
            # train the model
            model.fit(X_inner_train, y_inner_train, ids_inner_train)

            # evaluate
            X_inner_test = X_train[test_inner_index]
            y_inner_test = y_train[test_inner_index]
            ids_inner_test = ids_train[test_inner_index]

            # compute the score
            pred = model.predict(X_inner_test)
            # get per query scores
            query_groups = get_groups(ids_inner_test)

            scores = [metric.evaluate_preds(qid, y_inner_test[a:b], pred[a:b])
                      for qid, a, b in query_groups]

            print('score :' + str(scores))
            if str() in config_scores:
                config_scores[i].extend(scores)
            else:
                config_scores[i] = scores


    list_tuples = []

    for i in config_scores:
        mean_score = np.mean(config_scores[i])
        list_tuples.append((grid[i], mean_score))

    # return a list of tuples
    # (config_dict, score)

    return list_tuples


def extract_docs_for_rels(qrels):
    trec_ids = qrels['trec_ids']
    # split into ten chunks
    for i, chunk in enumerate(chunk_list(trec_ids,10)):
        print('Chunk ' + str(i))
        retrieve_docs_from_cache(chunk, cache_path='doc_cache', verbose=True, n_cores=4)

