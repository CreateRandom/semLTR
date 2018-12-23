import numpy as np
import scipy.stats as stats
import Orange
import matplotlib.pyplot as plt
import scikit_posthocs as posthocs

def perform_friedman_nemenyi(query_scores, condition_names):

    chi, p = stats.friedmanchisquare(*query_scores)

    print('p ' + str(p))

    # transpose such that queries are the columns
    condition_scores = np.transpose((np.array(query_scores) * -1))

    ranks = np.zeros(condition_scores.shape)
    for i, query in enumerate(condition_scores):
        # rank the conditions on this query
        ranks[i] = stats.rankdata(query)

    average_ranks = np.average(ranks, axis=0)

    print('Average ranks')


    for i, rank in enumerate(average_ranks):
        print(str(condition_names[i]) + ': ' + str(rank))

    # look into https://stackoverflow.com/questions/43383144/how-can-plot-results-of-the-friedman-nemenyi-test-using-python

    sigs = None

    # if Friedman was significant, do the post-test
    if p < 0.05:
        n_queries = len(condition_scores)
        # critical distance of Nemenyi post-test
        cd = Orange.evaluation.compute_CD(average_ranks, n_queries)  # tested on 30 datasets
        # plot the Nemenyi test results
        Orange.evaluation.graph_ranks(average_ranks, condition_names, cd=cd, width=6, textspace=1.5)
        plt.show()

        # transpose the scores such that queries are the columns
        sigs = posthocs.posthoc_nemenyi_friedman(condition_scores)

    return p, sigs


def analyze_results(results):

    for condition in results:
        print(condition)
        print(results[condition])

    query_pure_means = {}

    query_scores_reranked = []
    condition_names = []
    mean_scores_by_condition_reranked = {}
    all_qids = []
    for condition in results:
        condition_names.append(condition)
        # list of dicts, one per fold
        all_scores_ranked = results[condition]['all_scores_ranked']
        merged = {}
        for score_dict in all_scores_ranked:
            merged = {**merged,**score_dict}
        qids = list(merged.keys())
        # just plain scores
        scores = list(merged.values())
        all_qids.append(qids)
        query_scores_reranked.append(scores)
        mean_score = np.mean(scores)
        mean_scores_by_condition_reranked[condition] = mean_score

        fold_means_model = results[condition]['fold_means_model']
        query_pure_means[condition] = np.mean(fold_means_model)



    print('Mean re-ranked NDCG')
    print(str(mean_scores_by_condition_reranked))

    print('Mean pure NDCG')
    print(str(query_pure_means))

    # for each condition, compute the gains / losses with respect to the baseline

    gain_scores = np.array(query_scores_reranked)
    # treat the first line as baseline
    gain_scores = gain_scores - gain_scores[0]

    condition_gains = {}

    for i, gains in enumerate(gain_scores):
        condition_name = condition_names[i]
        # count number of improvements
        n_gains = 0
        n_losses = 0
        for gain in gains:
            if gain > 0:
                n_gains += 1
            else:
                n_losses += 1

        condition_gains[condition_name] = (n_gains, n_losses)

    # now, we know query by query gains

    # significance tests on the ranks
    p,sigs = perform_friedman_nemenyi(query_scores_reranked,condition_names)

    return {'p' : p, 'sigs' : sigs, 'query_pure_means' : query_pure_means,
            'query_scores_reranked' : query_scores_reranked,
            'mean_scores_by_condition_reranked': mean_scores_by_condition_reranked}