import pandas as pd
from gensim.models import KeyedVectors
from collections import Counter
from extraction.document_extraction import Document
from extraction.lmir import LMIR
import numpy as np
from nltk import ngrams
from nltk.tokenize import word_tokenize
import spacy

class FeaturePipeline(object):

    # dynamically decide which extractors to run
    # by passing them via the constructor
    def __init__(self, extractors):
        # a list of extractors to be applied sequentially
        self.extractors = extractors

    def extract_all_features(self, query, documents,**kwargs):
        """
            :param query: a query string
            :param documents: a list of docs, each of which is a Document
        """

        # A dictionary from field_feature to lists of document values
        # e.g. {body_wordcount: [], title_wordcount[]}
        feature_value_dict = {}

        for extractor in self.extractors:
            temp_dict = extractor.get_all_features(query,documents,**kwargs)
            feature_value_dict = {**feature_value_dict, **temp_dict}

        return pd.DataFrame(data=feature_value_dict)

# An abstract class for feature extraction functionality
class FeatureExtractor(object):

    def get_all_features(self, query, documents,**kwargs):
        """
        :param query: a query string
        :param documents: a list of document objects
        :return: dict of lists
        """
        LD = []
        # iterate over all docs
        for document in documents:
            all_feature_field_combinations = {}
            # and over all field
            for field in document.field_dict.keys():
                # for this field in this doc, e.g. title in document 1022
                # compute all functions
                for function in self.get_extraction_functions():
                    # a dict that has all feature_field names as keys and the feature values as values
                    feature_dict = function(query, field, document.field_dict,**kwargs)
                    # add this to the list of all these field dicts for this document
                    all_feature_field_combinations = {**all_feature_field_combinations, **feature_dict}
            # add to the storage of all doc_field_features
            LD.append(all_feature_field_combinations)

        # change this into a dict of lists
        # key: feature name
        # value: list of values
        v = {k: [dic[k] for dic in LD] for k in LD[0]}

        return v

    def get_extraction_functions(self) -> list:
        """
            Returns a dict of the form: {name : function}
            Names will be used as labels and the function will be
            called to compute features
            functions should take the following arguments
            extraction_function(query, text)
        """
        pass


# A simple dummy feature extractor to serve as an example
# Feature computed: Binary, query present in text?
class QueryPresenceExtractor(FeatureExtractor):

    def get_extraction_functions(self):
        return [self.queryPresent]

    def queryPresent(self, query, field, field_dict,**kwargs) -> dict:
        """
        :param query:  a query represented as string
        :param text: a text represented as string
        :return: 1 if query in text, 0 otherwise
        """

        text = field_dict[field]['text']

        queryPresent = 1 if query in text else 0

        query_text_ratio = len(query) / (len(text) + 1)

        return {str(field) + '_queryPresent' : queryPresent,
                str(field) + '_queryTextRatio': query_text_ratio}



# Extractor to extract the number of chars / words in a text section
class DocumentLengthExtractor(FeatureExtractor):
    def get_extraction_functions(self):
        return [self.computeLength]

    def computeLength(self,query,field,field_dict,**kwargs):
        nChars = len(field_dict[field]['text'])
        nWords =len(field_dict[field]['tokenized_text'])
        return {
            str(field) + '_nChars' : nChars,
            str(field) + '_nWords' : nWords,
        }

class LMIRExtractor(FeatureExtractor):

    def get_extraction_functions(self):
        return [self.compute_lmir]

    def compute_lmir(self,query,field,field_dict,**kwargs):
        text = field_dict[field]['tokenized_text']
        query = word_tokenize(query)
        doc_model = LMIR([text])

        # different smoothing methods
        abs_value = doc_model.absolute_discount(query)[0]
        dir_value = doc_model.dirichlet(query)[0]
        jm_value = doc_model.jelinek_mercer(query)[0]

        return {
            str(field) + '_lmirAbs' : abs_value,
            str(field) + '_lmirDir' : dir_value,
            str(field) + '_lmirJm' : jm_value
        }

class W2VExtractor(FeatureExtractor):

    def __init__(self, w2v_model) -> None:
        self.w2v_model = w2v_model

    def get_extraction_functions(self) -> list:
        return [self.compute_w2v]

    def compute_w2v(self,query,field,field_dict,**kwargs):
        """
            Mean similarity of query terms and documents +
            Max similarity of query terms and documents
        """

        query = word_tokenize(query)
        text = field_dict[field]['tokenized_text']
        sims = []
        for q_word in query:
            for t_word in text:
                if q_word in self.w2v_model and t_word in self.w2v_model:
                    sims.append(self.w2v_model.similarity(q_word, t_word))

        # check to see whether we found any similarities
        mean_score = 0 if not sims else np.mean(sims)
        max_score = 0 if not sims else np.max(sims)

        return {
            str(field) + '_w2vMean' : mean_score,
            str(field) + '_w2vMax' : max_score
        }


# Extractor to extract BM25F
class BM25Extractor(FeatureExtractor):

    def get_extraction_functions(self):
        return [self.compute_BM25_features]


    def compute_BM25_features(self,query,field,field_dict,**kwargs):
        #                       PREPARING DATA
        # First, get constants from the corpus (like e.g. idf values):
        idf_values = kwargs['idf_values']
        if field == 'title':
            average_length = kwargs['mean_title_length']
        if field == 'body':
            average_length = kwargs['mean_body_length']
        # mean_body_length = kwargs['mean_body_length']
        # mean_title_length = kwargs['mean_title_length']

        tokenized_Doc = field_dict[field]['tokenized_text']
        # Now, tokenize the query
        tokenized_query = word_tokenize(query.lower())
        # Then, count all individual words in the document:
        all_document_words_counts = Counter(tokenized_Doc)
        individual_doc_words = list(all_document_words_counts.keys())
        # Compute the document text's length:
        document_text_length = len(tokenized_Doc)
        #                        INITIALIZING FEATURES
        sum_query_word_occurrences = 0
        sum_query_word_tfidf_values = 0
        sum_idf_values = 0
        score_BM25 = 0
        k_1 = 1.2
        b = 0.75
        #                        COMPUTING FEATURES
        for query_word in tokenized_query:
            if field in idf_values[query_word]:
                query_word_idf = idf_values[query_word][field]
                sum_idf_values = sum_idf_values + query_word_idf
            else:
                query_word_idf = 0
            # Scores will only change if the current query_word occurs in the document:
            if query_word in individual_doc_words:
                # First, compute how often the current query word occurs in the document:
                query_word_term_frequency = all_document_words_counts[query_word]
                # -> tfidf:
                query_word_tfidf = query_word_idf*query_word_term_frequency
                # Update the sum of query_words' occurrences in the document:
                sum_query_word_occurrences = sum_query_word_occurrences+query_word_term_frequency
                # Update the sum over all query_words' tfidf values:
                sum_query_word_tfidf_values = sum_query_word_tfidf_values + query_word_tfidf
                # Update the BM25 value:
                score_BM25 = score_BM25 + query_word_idf * query_word_term_frequency*(k_1+1)/(query_word_term_frequency+k_1*(1-b+b*document_text_length/average_length))

        return {
            str(field) + '_sum_query_word_occurrences' : sum_query_word_occurrences,
            str(field) + '_sum_query_word_tfidf_values': sum_query_word_tfidf_values,
            str(field) + '_sum_idf_values': sum_idf_values,
            str(field) + '_BM25': score_BM25
        }


class SubjObjRelationExtractor(FeatureExtractor):

    def get_extraction_functions(self):
        return [self.compute_subj_obj_relation_features]

    def compute_subj_obj_relation_features(self,query,field,field_dict,**kwargs):
        # Extract all counts from our document:
        no_subjs = field_dict[field]['no_subjs']
        no_iobjs = field_dict[field]['no_iobjs']
        no_dobjs = field_dict[field]['no_dobjs']
        no_rootvs = field_dict[field]['no_rootvs']
        no_subj_chunks = field_dict[field]['no_subj_chunks']
        no_iobj_chunks = field_dict[field]['no_iobj_chunks']
        no_dobj_chunks = field_dict[field]['no_dobj_chunks']

        # Generate all n-grams up to six from our query
        query_unigrams = query.split()
        query_bigrams = [' '.join(ngram) for ngram in ngrams(query_unigrams,2)]
        query_trigrams = [' '.join(ngram) for ngram in ngrams(query_unigrams, 3)]

        # Count how often each query unigram appears as a subject etc in the document's text:
        sum_unigram_as_subj = 0
        sum_unigram_as_iobj = 0
        sum_unigram_as_dobj = 0
        sum_unigram_as_rootv = 0
        for query_unigram in query_unigrams:
            sum_unigram_as_subj     = sum_unigram_as_subj  + no_subjs[query_unigram]
            sum_unigram_as_iobj     = sum_unigram_as_iobj  + no_iobjs[query_unigram]
            sum_unigram_as_dobj     = sum_unigram_as_dobj  + no_dobjs[query_unigram]
            sum_unigram_as_rootv    = sum_unigram_as_rootv + no_rootvs[query_unigram]
        del query_unigram

        # Count how often each query bigram appears as a subject chunk etc in the document's text:
        sum_bigram_as_subj,sum_bigram_as_iobj, sum_bigram_as_dobj \
            = self.count_ngrams_in_counted_noun_chunks(query_bigrams,no_subj_chunks,no_iobj_chunks,no_dobj_chunks)
        # Count how often each query trigram appears as a subject chunk etc in the document's text:
        sum_trigram_as_subj,sum_trigram_as_iobj, sum_trigram_as_dobj \
            = self.count_ngrams_in_counted_noun_chunks(query_trigrams,no_subj_chunks,no_iobj_chunks,no_dobj_chunks)

        return{
            str(field) + '_sum_unigram_as_subj': sum_unigram_as_subj,
            str(field) + '_sum_unigram_as_iobj': sum_unigram_as_iobj,
            str(field) + '_sum_unigram_as_dobj': sum_unigram_as_dobj,
            str(field) + '_sum_unigram_as_rootv': sum_unigram_as_rootv,
            str(field) + '_sum_bigram_as_subj': sum_bigram_as_subj,
            str(field) + '_sum_bigram_as_iobj': sum_bigram_as_iobj,
            str(field) + '_sum_bigram_as_dobj': sum_bigram_as_dobj,
            str(field) + '_sum_trigram_as_subj': sum_trigram_as_subj,
            str(field) + '_sum_trigram_as_iobj': sum_trigram_as_iobj,
            str(field) + '_sum_trigram_as_dobj': sum_trigram_as_dobj,
        }



    def count_ngrams_in_counted_noun_chunks(self, query_ngrams, counted_subj_chunk, counted_iobj_chunk, counted_dobj_chunk):
        sum_ngram_as_subj_chunk = 0
        sum_ngram_as_iobj_chunk = 0
        sum_ngram_as_dobj_chunk = 0
        for query_ngram in query_ngrams:
            sum_ngram_as_subj_chunk = sum_ngram_as_subj_chunk + counted_subj_chunk[query_ngram]
            sum_ngram_as_iobj_chunk = sum_ngram_as_iobj_chunk + counted_iobj_chunk[query_ngram]
            sum_ngram_as_dobj_chunk = sum_ngram_as_dobj_chunk + counted_dobj_chunk[query_ngram]
        return sum_ngram_as_subj_chunk, sum_ngram_as_iobj_chunk, sum_ngram_as_dobj_chunk



# TODO: preprocessing?-> lowercase?

# TODO: lemmatization instead of tokenization?

# TODO: sumIDFQueryTermsInDoc​
