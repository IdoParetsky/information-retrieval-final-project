from flask import Flask, request, jsonify
import math
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from inverted_index import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
spark = SparkSession.builder.getOrCreate()

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')


    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    idx_body = InvertedIndex.read_index('body_index', 'body')
    words, pls = zip(*idx_body.posting_lists_iter())
    tok_query = tokenize(query)
    Q = generate_query_tfidf_vector(tok_query, idx_body)
    D = generate_document_tfidf_matrix(tok_query, idx_body, words, pls)
    cos_sim = cosine_similarity(D, Q)
    res = get_top_n(cos_sim, 100)

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    #query = request.args.get('query', '')
    query = ' '
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    idx_title = InvertedIndex.read_index('.', 'title_index')
    words, pls = zip(*idx_title.posting_lists_iter())
    tok_query = tokenize(query)
    candidates_distinct = {}
    for term in np.unique(tok_query):
        if term in words:
            list_of_doc = pls[words.index(term)]
            for doc_id, freq in list_of_doc:
                if doc_id not in candidates_distinct.keys():
                    candidates_distinct[doc_id] = 1
                else:
                    candidates_distinct[doc_id] += 1
    sorted_candidates = dict(sorted(candidates_distinct.items(), key=lambda item: item[1]))
    res = []
    for id, count in sorted_candidates.items():
        res.append(id)

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    #query = request.args.get('query', '')
    query  = ' '
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    idx_anchor = InvertedIndex.read_index('.', 'anchor_index')
    print(idx_anchor.posting_locs)

    words, pls = zip(*idx_anchor.posting_lists_iter())
    tok_query = tokenize(query)
    candidates_distinct = {}
    for term in np.unique(tok_query):
        if term in words:
            list_of_doc = pls[words.index(term)]
            for doc_id, freq in list_of_doc:
                if doc_id not in candidates_distinct.keys():
                    candidates_distinct[doc_id] = 1
                else:
                    candidates_distinct[doc_id] += 1
    sorted_candidates = dict(sorted(candidates_distinct.items(), key=lambda item: item[1],reverse=True))
    res = []
    for id, count in sorted_candidates.items():
        res.append(id)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


english_stopwords = frozenset(stopwords.words('english'))
# We queried ChatGPT for Wikipedia-specific StopWords and added some of our own
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "first", "see", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became", "page", "article",
                    "reference", "source", "content",
                    "fact", "year", "date", "place", "wiki",
                    "edit", "version", "user", "talk", "discussion", "template",
                    "category", "project", "author", "writer",
                    "creator", "publisher", "editor", "publication", "edition",
                    "issue", "chapter"]
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens

def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [(doc_id, (freq / index.DL[str(doc_id)]) * math.log(len(index.DL) / index.df[term], 10)) for
                               doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates

def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.


    words,pls: iterator for working with posting.

    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(index.term_total)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,
                                                           pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)
    D.index = unique_candidates
    D.columns = index.term_total.keys()
    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    D = spark.createDataFrame(D)
    return D

def generate_query_tfidf_vector(query_to_search, index):

    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q

def get_top_n(sim_dict, N=3):
        """
        Sort and return the highest N documents according to the cosine similarity score.
        Generate a dictionary of cosine similarity scores

        Parameters:
        -----------
        sim_dict: a dictionary of similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

        N: Integer (how many documents to retrieve). By default N = 3

        Returns:
        -----------
        a ranked list of pairs (doc_id, score) in the length of N.
        """

        return sorted([(doc_id, round(score, 10)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                      reverse=True)[:N]

def cos_sim_formula(doc, query):
    numerator = sum(doc * query)
    denumerator = ((query ** 2).sum() ** 0.5) * ((doc ** 2).sum() ** 0.5)
    return numerator / denumerator

def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """

    res = {}
    for idx, row in D.iterrows():
        res[idx] = cos_sim_formula(row, Q)
    return res

is_gcp = False
# if not is_gcp:
#     # Copy one wikidumps files
#     import os
#     from pathlib import Path
#
#     ## RENAME the project_id to yours project id from the project you created in GCP
#     project_id = 'crypto-lexicon-370515'
#     !gcloud
#     config
#     set
#     project
#     {project_id}
#
#     data_bucket_name = 'wikidata20210801_preprocessed'
#     try:
#         if os.environ["wikidata20210801_preprocessed"] is not None:
#             pass
#     except:
#         !mkdir
#         wikidumps
#         !gsutil - u
#         {project_id}
#         cp
#         gs: // {data_bucket_name} / multistream1_preprocessed.parquet
#         "wikidumps/"
#
#     try:
#         if os.environ["wikidata20210801_preprocessed"] is not None:
#             path = os.environ["wikidata20210801_preprocessed"] + "/wikidumps/*"
#     except:
#         path = "wikidumps/*"
#
#     parquetFile = spark.read.parquet(path)
#     # take the 'title', 'text', 'anchor_text' and 'id' or the first 1000 rows and create an RDD from it
#     doc_title_text_anchor_quadruplets = parquetFile.limit(1000).select("title", "text", "anchor_text", "id").rdd
#
#     from inverted_index_colab import *
#
#     id_title = spark.read.parquet(path).limit(1000).select("id", "title")
#
# else:
#     # Put your bucket name below and make sure you can access it without an error
#     bucket_name = '318419512_318510252'
#     full_path = f"gs://{318419512_318510252}/"
#     paths = []
#
#     client = storage.Client()
#     blobs = client.list_blobs(bucket_name)
#     for b in blobs:
#         if b.name != 'graphframes.sh':
#             paths.append(full_path + b.name)
#     parquetFile = spark.read.parquet(*paths)
#     from inverted_index_gcp import InvertedIndex
#
#     id_title = parquetFile.select("id", "title").rdd
#
# import pyspark.sql.functions as f
# get_title_from_id = lambda wiki_id: id_title.filter(f.col('id')==wiki_id).collect()[0][1]

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    #app.run(host='0.0.0.0', port=8080, debug=True)
    search_anchor()

