from flask import Flask, request, jsonify
import math
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from inverted_index_colab import *
from pyspark.sql import SparkSession
import time
import math



class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search(search_query=None):
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
    
    if search_query:
        query = search_query
    else:
        query = request.args.get('query', '')
    
    if len(query) == 0:
      return [] if search_query else jsonify([])
    # BEGIN SOLUTION
    res1 = search_body(query)
    print(f"{res1 = }")
    res2 = search_title(query)[:100]
    print(f"{res2 = }")
    res3 = search_anchor(query)[:100]
    print(f"{res3 = }")
    ids_of_res1 = [res1[0] for item in res1]
    ids_of_res2 = [res2[0] for item in res2]
    ids_of_res3 = [res3[0] for item in res3]
    combined_list = res1 + res2 + res3
    counts = Counter(combined_list)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    res = [item[0] for item in sorted_counts[:100]]

    # END SOLUTION
    return res
    return jsonify(res)

@app.route("/search_body")
def search_body(search_query=None):
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
    
    if search_query:
        query = search_query
    else:
        query = request.args.get('query', '')

    # BEGIN SOLUTION

    tok_query = tokenize(query)
    filitered_query = list(filter(lambda x: x in body_words, tok_query))
    query_tf = {}
    query_tf = {word: query_tf.get(word, 0) + 1 for word in filitered_query}
    query_tf_idf =  {word: tf / len(tok_query) * math.log10(N / idx_body.df[word]) for word, tf in query_tf.items()}
    
    candidates_tf_idf = {}
    candidates_tf_idf = {(doc_id, word): candidates_tf_idf.get((doc_id, word), 0) + (freq / idx_body.DL[doc_id]) * math.log10(N / idx_body.df[word]) for word in filitered_query for doc_id, freq in body_pls[body_words.index(word)]}
    
    print(f"{filitered_query = }")
    print(f"{candidates_tf_idf = }")
    #doc_id_list = list(set()
    cos_sim_scores = {}
    #for (doc_id, ) in candidates_tf_idf:
        
    
    #cos_sim_dict = {doc_id: for (doc_id, word), doc_tfidf in candidates.items() for word in filitered_query}
    
    
    #Q = generate_query_tfidf_vector(tok_query, body_words, body_pls)
    #D = generate_document_tfidf_matrix(tok_query, idx_body, body_words, body_pls)
    #cos_sim = cosine_similarity(D, Q)
    res = get_top_n(cos_sim, 100)

    return [(key, id_title_pr_pv_dict[str(key)][0]) for key, score in res] 
    
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title(search_query=None):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 
        http://be08-35-196-42-90.ngrok.io/search_title?query=hello+world
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).

    '''

    if search_query:
        query = search_query
    else:
        query = request.args.get('query', '')
    
    if len(query) == 0:
      return [] if search_query else jsonify([])
    # BEGIN SOLUTION

    tok_query = set(tokenize(query))
    candidates_distinct = {}
    """
    for term in tok_query:
        for doc_id, _ in title_words_pls.get(term, (None, None)):
            if doc_id:
                doc_pagerank = id_title_pr_pv_dict[str(doc_id)][1]  # take the matching pagerank to the given id from the global dict 
                candidates_distinct[doc_id] = (candidates_distinct.get(doc_id, (0, doc_pagerank))[0] + 1, doc_pagerank)
    """
    filitered_query = list(filter(lambda x: x in title_words_pls, tok_query))
    candidates_distinct = {doc_id: candidates_distinct.get(doc_id, 0) + 1 for word in filitered_query for (doc_id, _) in title_words_pls[word]}

    candidates_distinct = sorted({doc_id: (cnt, *id_title_pr_pv_dict[str(doc_id)][1::-1]) for doc_id, cnt in  candidates_distinct.items()}.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)

    res = [(doc_id, cnt_pr_title[2]) for doc_id, cnt_pr_title in candidates_distinct]
    return res if search_query else jsonify(res)

    # END SOLUTION

@app.route("/search_anchor")
def search_anchor(search_query=None):
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
    
    if search_query:
        query = search_query
    else:
        query = request.args.get('query', '')
    
    if len(query) == 0:
      return [] if search_query else jsonify([])
    # BEGIN SOLUTION

    tok_query = tokenize(query)
    candidates_distinct = {}
    """
    for term in np.unique(tok_query):
        if term in anchor_words_pls.keys():
            list_of_docs = anchor_words_pls[term]
            for doc_id in list_of_docs:
                doc_pagerank = id_title_pr_pv_dict[str(doc_id[0])][1] 
                if doc_id[0] in candidates_distinct.keys():
                    candidates_distinct[doc_id[0]] = (candidates_distinct[doc_id[0]][0]+1,doc_pagerank)
                else:
                    candidates_distinct[doc_id[0]] = (1, doc_pagerank)
        else:
            print(f"The word '{term}' in query '{query}' does NOT appear in our Anchor Index")
    
    
    candidates_distinct = sorted(candidates_distinct.items(), key=lambda x: (x[1][0], x[1][1]),reverse=True)
    return [(key[0], id_title_pr_pv_dict[str(key[0])][0]) for key in candidates_distinct] 
    """
    filitered_query = list(filter(lambda x: x in anchor_words_pls, tok_query))
    candidates_distinct = {doc_id: candidates_distinct.get(doc_id, 0) + 1 for word in filitered_query for (doc_id, _) in anchor_words_pls[word]}
    
    candidates_distinct = sorted({doc_id: (cnt, *id_title_pr_pv_dict[str(doc_id)][1::-1]) for doc_id, cnt in  candidates_distinct.items()}.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)

    res = [(doc_id, cnt_pr_title[2]) for doc_id, cnt_pr_title in candidates_distinct]
    return res if search_query else jsonify(res)
    # END SOLUTION

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
    #wiki_ids = request.get_json()
    wiki_ids = [12,25,999]
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for id in wiki_ids:
        if str(id) in id_title_pr_pv_dict.keys():
            res.append((id_title_pr_pv_dict[str(id)][0],id_title_pr_pv_dict[str(id)][1]))
        else:
            print("Unknown ID")
    
    # END SOLUTION
    return res
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
    wiki_ids = [12,25,999]
    #wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    for id in wiki_ids:
        if str(id) in id_title_pr_pv_dict.keys():
            res.append((id_title_pr_pv_dict[str(id)][0],id_title_pr_pv_dict[str(id)][2]))
        else:
            print("Unknown ID")
    
    
    return res
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
            normlized_tfidf = [(doc_id, (freq / index.DL[doc_id]) * math.log(len(index.DL) / index.df[term], 10)) for
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

    total_vocab_size = len(words)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,
                                                           pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = words

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf
    return D

def generate_query_tfidf_vector(query_to_search, words, pls):

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
    
    
    Q = np.zeros(len(body_words))
    counter = Counter(query_to_search)
    for token in query_to_search:
        if token in words:  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = idx_body.df[token]

            idf = math.log((len(idx_body.DL)) / (df + epsilon), 10)  # smoothing
            try:
                ind = body_words.index(token)
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

def load_data(load_data_dict):
    # load the three indexes
    global idx_body
    idx_body = InvertedIndex.read_index('./colab_indexes', 'body_index')
    idx_title = InvertedIndex.read_index('./colab_indexes', 'title_index')
    idx_anchor = InvertedIndex.read_index('./colab_indexes', 'anchor_index')

    # declare the variables we will need in other functions (search) as global
    global body_words
    global body_pls
    global title_words_pls
    global anchor_words_pls
    global id_pagerank_title_pageviews_df
    global id_title_pr_pv_dict
    global N  # Number of documents in the corpus
    
    # make a dictionary such as {word1: pls1, word2: pls2 ...}

    body_words, body_pls = zip(*idx_body.posting_lists_iter())
    title_words_pls = dict(zip(*zip(*idx_title.posting_lists_iter())))
    anchor_words_pls = dict(zip(*zip(*idx_anchor.posting_lists_iter())))
    id_title_pr_pv_dict = load_data_dict 
    N = len(idx_body.DL)

    

    
    
