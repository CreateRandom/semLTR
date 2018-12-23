import multiprocessing
import os
import re
import pickle
import spacy
import time
from functools import partial
from bs4 import BeautifulSoup
from collections import Counter
from chatnoir import chatnoir_api as api
from nltk.tokenize import word_tokenize
from util.general_util import chunk_list, merge_dict_list

nlp = spacy.load('en')

class Document(object):
    # Simple container class
    # dict: {'title': 'bla','body': 'bla'}
    # link : Not currently implemented
    def __init__(self, field_dict, link):
        self.field_dict = field_dict
        self.link = link


def extract_doc_from_html(html):
    """
    Parses an html file to differentiate between individual parts in the underlying document.
    :param html:    The html file to be parsed
    :return: A Document object with a a field dict (that contains the documents' different parts) and a link.
    """
    # Apply BeautifulSoup to the html
    soup = BeautifulSoup(html, features="html.parser")
    # Now, we can easily extract the title and the body of the html:
    title_obj = soup.find('title')
    title = ' ' if title_obj is None else title_obj.get_text(strip=True)
    title = preprocess_retrieved_document_text(title).lower()
    # Body
    body_obj = soup.find('body')
    body = ' ' if body_obj is None else body_obj.get_text(strip=True)
    body = preprocess_retrieved_document_text(body).lower()


    # Also, tokenize both now:
    tokenized_title = word_tokenize(title.lower())
    tokenized_body = word_tokenize(body.lower())

    # Now, we want subjects, indirect objects, objects and verbs inside this text.
    # We need them in a counted form using Counter.
    # Also, we need counts for chunks of words that have these functions in a sentence:
 #   nlp = spacy.load('en')
    no_subjs_title,no_iobjs_title,no_dobjs_title,no_rootvs_title,no_subj_chunks_title,no_iobj_chunks_title,no_dobj_chunks_title = \
        get_counted_grammar_entities(nlp,title)
    no_subjs_body,no_iobjs_body,no_dobjs_body,no_rootvs_body,no_subj_chunks_body,no_iobj_chunks_body,no_dobj_chunks_body = \
        get_counted_grammar_entities(nlp,body)

    return Document(field_dict=
                    {'title':
                         {'text':title,
                          'tokenized_text':tokenized_title,
                          'no_subjs':no_subjs_title,
                          'no_iobjs':no_iobjs_title,
                          'no_dobjs':no_dobjs_title,
                          'no_rootvs':no_rootvs_title,
                          'no_subj_chunks':no_subj_chunks_title,
                          'no_iobj_chunks':no_iobj_chunks_title,
                          'no_dobj_chunks':no_dobj_chunks_title,
                          },
                    'body':
                         {'text':body,
                          'tokenized_text':tokenized_body,
                          'no_subjs':no_subjs_body,
                          'no_iobjs':no_iobjs_body,
                          'no_dobjs':no_dobjs_body,
                          'no_rootvs':no_rootvs_body,
                          'no_subj_chunks':no_subj_chunks_body,
                          'no_iobj_chunks':no_iobj_chunks_body,
                          'no_dobj_chunks':no_dobj_chunks_body,
                          }
                    }, link=None)

def preprocess_retrieved_document_text(document_text):
    preprocessed_text = document_text.replace('\n', ' ').replace('\t', '')
    preprocessed_text = re.sub(r' +',' ', preprocessed_text)
    preprocessed_text = re.sub(r"[^\w\d'\.,!\?\s]+", '', preprocessed_text)
    return preprocessed_text

def get_counted_grammar_entities(nlp,document_text):
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                  "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
                  "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
                  "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
                  "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
                  "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
                  "in",
                  "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
                  "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
                  "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
                  "will", "just", "don", "should", "now"]
    # Get our document text and parse it with spacy
    #nlp = spacy.load('en')
    # spacy complains for more than a million chars, so cut this document
    document_text = document_text[:999999]
    parsed_text = nlp(document_text)

    # First, we will only regard single word inside the document's text.
    # Look for all subjects, indirect objects and root verbs:
    doc_subjs = []
    doc_iobjs = []
    doc_dobjs = []
    doc_rootvs = []
    for text in parsed_text:
        # aa = text.dep_ # TODO: REMOVE
        # children = [child for child in text.children if child.dep_ == "compound"]
        # subject
        if text.dep_ == "nsubj":
            # subject = text.orth_
            doc_subjs.append(text.orth_)
        # indirect object
        if text.dep_ == "iobj":
            # indirect_object = text.orth_
            doc_iobjs.append(text.orth_)
        # direct object
        if text.dep_ == "dobj":
            # direct_object = text.orth_
            doc_dobjs.append(text.orth_)
        # root verb
        if text.dep_ == "ROOT":
            doc_rootvs.append(text.orth_)

    # Next, look at chunks of words:
    doc_chunks = []
    doc_subj_chunks = []
    doc_iobj_chunks = []
    doc_dobj_chunks = []
    for chunk in parsed_text.noun_chunks:
        # Determiners etc usually do not occur in queries:
        chunk_without_stopwords = ' '.join([word.text for word in chunk if not (word.text.lower() in stop_words)])
        if not chunk_without_stopwords == '':  # Only consider chunks not solely consisting of stopwords
            if chunk.root.dep_ == "nsubj":
                doc_subj_chunks.append(
                    chunk_without_stopwords)  # [chunk.text, chunk_without_stopwords, chunk.root.text, chunk.root.head.text])
            if chunk.root.dep_ == "dobj":
                doc_dobj_chunks.append(
                    chunk_without_stopwords)  # [chunk.text, chunk_without_stopwords, chunk.root.text, chunk.root.head.text])
            if chunk.root.dep_ == "iobj":
                doc_iobj_chunks.append(
                    chunk_without_stopwords)  # [chunk.text, chunk_without_stopwords, chunk.root.text, chunk.root.head.text])

        doc_chunks.append([chunk.text, chunk.root.text, chunk.root.dep_,
                           chunk.root.head.text])
        del chunk_without_stopwords, chunk

    # Now, count all of these occurences:
    no_subjs = Counter(doc_subjs)
    no_iobjs = Counter(doc_iobjs)
    no_dobjs = Counter(doc_dobjs)
    no_rootvs = Counter(doc_rootvs)
    no_subj_chunks = Counter(doc_subj_chunks)
    no_iobj_chunks = Counter(doc_iobj_chunks)
    no_dobj_chunks = Counter(doc_dobj_chunks)

    return no_subjs, no_iobjs, no_dobjs, no_rootvs, no_subj_chunks, no_iobj_chunks, no_dobj_chunks


def retrieve_docs_from_cache(list_trec_ids, cache_path,verbose=False, n_cores=-1):
    """
    Retrieves Document for each trec id in the list, relying on a cache specified in the path
    if the cache doesn't exist yet, it'll be created and content will be retrieved from the web
    and added.
    :param list_trec_ids: List of TREC ids for each of which a document is retrieved
    :param cache_path: The path to cache to
    :return: A dictionary from ID to document objects
    """
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

    # find all ids we haven't stored yet
    missing = [id for id in list_trec_ids if id not in index_dict]
    # if stuff is missing, grab it from the web and store it in the dict
    if len(missing) > 0:
        new_id_content = retrieve_docs(missing, verbose=verbose,n_cores=n_cores)
        # write the new content to the disk file by file
        for id in new_id_content:
            file_name = id +'.p'
            file_path = os.path.join(cache_path, file_name)
            with open(file_path,'wb') as pfile:
                pickle.dump(new_id_content[id],pfile)
            # only store the name, not the full doc in the dict
            new_id_content[id] = file_name
        # merge dictionaries
        index_dict = {**index_dict, **new_id_content}

    # dump the index
    with open(index_path, 'wb') as pfile:
        pickle.dump(index_dict, pfile)

    trec_to_doc_dict = {}
    # only load the files we need
    for id in list_trec_ids:
        file_name = index_dict[id]
        file_path = os.path.join(cache_path, file_name)
        with open(file_path, 'rb') as pfile:
            # load up the dict
            doc = pickle.load(pfile)
            trec_to_doc_dict[id] = doc

    return trec_to_doc_dict


def retrieve_docs(list_trec_ids, verbose=False,n_cores=-1):
    """
    Multi-process web retrieval
    based on: https://www.eamonnbell.com/blog/2015/10/05/the-right-way-to-use-requests-in-parallel-in-python/

    :param list_trec_ids: A list of IDs for which documents will be retrieved
    :return: A dictionary from IDs to Document objects
    """

    if n_cores is 1:
        pool_outputs = fetch_docs_from_web(list_trec_ids,verbose=verbose)
    # multiple processes
    else:
        # if the number is unspecified
        if n_cores is -1:
            # one process per core
            n_cores = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(processes=n_cores)

        # split into a list of lists, one per process
        chunked = chunk_list(list_trec_ids,n_cores)

        # multi-processing, calls up the function for fetching documents
        # we get a list of dicts back (one per process)
        pool_outputs = pool.map(partial(fetch_docs_from_web,verbose=verbose),
                                chunked)
        # get rid of the pool
        pool.close()
        pool.join()

        # merge the results
        pool_outputs = merge_dict_list(pool_outputs)

    return pool_outputs


def fetch_docs_from_web(trec_id_list, corpus='cw09', verbose=False):
    """
    Retrieves the HTML from the web and then extracts a document
    :param trec_id_list: A list of IDs to be retrieved
    :param corpus: the string name of the corpus
    :return: A dictionary from IDs to Document objects
    """
    id_to_html = api.retrieveHTML_from_TREC_IDs(trec_id_list, corpus=corpus, verbose=verbose)
    n_processed = 0
    start_time = time.time()
    # extract the document
    for id in id_to_html:
        id_to_html[id] = extract_doc_from_html(id_to_html[id])
        n_processed = n_processed + 1
        if verbose:
            if (n_processed > 0 and n_processed % 20 == 0):
                print('Processed ' + str(n_processed) + ' documents.')
                print("--- %s seconds needed ---" % (time.time() - start_time))
                start_time = time.time()

    return id_to_html
