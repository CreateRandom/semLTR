from chatnoir.chatnoir_api import retrieve_word_constants_from_chatnoir
import numpy as np



def extract_idfs_for_all_unique_query_terms(qd_pairs):
    """
    A function to extract the idf values for each unique word across all queries.
    Respective values will be stored.
    :param: qrel_dict: The qrels
    :return: unique_word_contants, which is a dictionary containing information like idf and average field length per section
             for each unique word across all queries
    """
    # To avoid redundancy, we want to extract the idf value only for all unique words across all queries.
    # Start: get query-document pairs (= a dict from a query id to other important values)
    # Concatenate the text of all queries:
    all_queries_text = " ".join([qd_pairs[query_id]['text'] for query_id in qd_pairs])
    # Now get all unique words across all queries:
    unique_words = set(all_queries_text.split(' '))

    # Create an empty dict to store the unique terms' information in
    unique_word_contants = {}
    # Now loop over all unique words and extract the respective idf values.
    for unique_word in unique_words:
        unique_word_contants[unique_word] = retrieve_word_constants_from_chatnoir(unique_word)
    del all_queries_text, qd_pairs

    # Compute the average over average field lengths:
    all_word_constants = list(unique_word_contants.values())
    title_lengths = []
    body_lengths = []
    for word_constants in all_word_constants:
        if 'title' in word_constants:
            title_lengths.append(word_constants['title']['avgFieldLength'])
        if 'body' in word_constants:
            body_lengths.append(word_constants['body']['avgFieldLength'])
    mean_title_length = np.mean(title_lengths)
    mean_body_length = np.mean(body_lengths)
    del title_lengths, body_lengths, all_word_constants, word_constants, unique_word

    # Flatten our information:
    idf_values = {}
    for unique_word in unique_words:
        idf_values[unique_word] = {}
        for field_name in unique_word_contants[unique_word].keys():
            if field_name in ['title','body']:
                idf_values[unique_word][field_name] = unique_word_contants[unique_word][field_name]['idf']

    del field_name, unique_word#, unique_word_contants


    return idf_values, mean_body_length, mean_title_length