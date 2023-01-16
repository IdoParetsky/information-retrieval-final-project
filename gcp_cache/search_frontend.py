from flask import Flask, request, jsonify
import math
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import defaultdict, Counter, ChainMap
from inverted_index_gcp import *
import time
import pickle
from pathlib import Path
from google.cloud import storage
import gc


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def search_tfidf_cos_sim(query, group, norm_cos_sim, is_search=False):
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE TITLE / BODY / ANCHOR TEXT of Wikipedia articles.

    Arguments:
    --------
        query: str
            Query String
        
        group: str
            "title" / "body" / "anchor"
        
        norm_cos_sim: bool
            Whether to divide cosine similarity by group's InvertedIndex.DL (in addition to TfIdf normalization)
        
        is_search: bool
            Was this function called from search function? (other option - search_body)
    
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title). 
        If this function was called from search function - return cos_sim_scores as well ({wiki_id: cos_sim_score})
    '''
    
    tok_query = tokenize(query)
    len_tok_query = len(tok_query)
    filtered_query = list(filter(lambda x: x in globals()[f"idx_{group}"].posting_locs, tok_query))  # Filter query to match words that appear in the group's posting list
    query_tf = Counter(filtered_query).items()
    query_tf_idf =  {word: tf / len(tok_query) * math.log10(N / globals()[f"idx_{group}"].df[word]) for word, tf in query_tf}
    
    candidates_tf_idf = {}
    for word, tf in query_tf:
        for wiki_id, freq in globals()[f"idx_{group}"].read_posting_list(word):

            candidates_tf_idf[wiki_id] = dict(candidates_tf_idf.get(wiki_id, {}), **{word:tf * freq * math.log10(N / globals()[f"idx_{group}"].df[word])})

    cos_sim_scores = get_top_n({wiki_id: sum([query_tf_idf[word] * candidates_tf_idf[wiki_id][word] / len_tok_query for word in candidates_tf_idf[wiki_id]]) for wiki_id in candidates_tf_idf}, 100)

    res = [(key, id_title_pr_pv_dict[key][0]) for key, _ in cos_sim_scores] 
    
    return res if not is_search else (res, dict(cos_sim_scores))  # if_search - return cos_sim_scores as well for scoring calculation


def search_boolean(query, group):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE / ANCHOR  TEXT (group) of Wikipedia articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in title / anchor text linking to the page. 
        
    Arguments:
    --------
        query: str
            Query String
        group: str
            "title" / "anchor"

    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    
    tok_query = tokenize(query)
    candidates_distinct = {}
    filtered_query = list(filter(lambda x: x in globals()[f"idx_{group}"].posting_locs, tok_query))  # Filter query to match words that appear in the group's posting list
    candidates_distinct = {wiki_id: candidates_distinct.get(wiki_id, 0) + 1 for word in filtered_query for (wiki_id, _) in globals()[f"idx_{group}"].read_posting_list(word)}
    
    candidates_distinct = sorted({wiki_id: (cnt, *id_title_pr_pv_dict[wiki_id][1::-1]) for wiki_id, cnt in  candidates_distinct.items()}.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)  # {wiki_id: (cnt, pagerank, title)}, sorted by cnt and then pagerank for tie-breaking
    
    res = [(wiki_id, cnt_pr_title[2]) for wiki_id, cnt_pr_title in candidates_distinct]
    
    return res


def normalize_result_set_pagerank_and_pageviews(res):
    ''' filters result set's PageRank and PageViews scores and normalizes them within the filtered result set.
        
     Arguments:
    --------
        res: list of (wiki_id: int, title: str)
            List of up to 100 search results, where each element is a tuple (wiki_id, title).
        
    Returns:
    --------
        2 dictionaries - {wiki_id: normalized PageRank score} and {wiki_id: normalized PageView score}
    '''
    
    wiki_id_pagerank_pageviews_dict = {wiki_id: id_title_pr_pv_dict[wiki_id][1:] for wiki_id, _ in res.items()}

    pagerank_array = np.array(list({wiki_id: pagerank_pageview[0] for wiki_id, pagerank_pageview in wiki_id_pagerank_pageviews_dict.items()}.items()), dtype=[('wiki_id', 'i4'), ('pagerank', 'f4')])
    pageview_array = np.array(list({wiki_id: pagerank_pageview[1] for wiki_id, pagerank_pageview in wiki_id_pagerank_pageviews_dict.items()}.items()), dtype=[('wiki_id', 'i4'), ('pageview', 'f4')])
    
    # Manual MinMax Scaling as a Vetctor Operation
    pagerank_array['pagerank'] = (pagerank_array['pagerank'] - pagerank_array['pagerank'].min()) / (pagerank_array['pagerank'].max() - pagerank_array['pagerank'].min())
    pageview_array['pageview'] = (pageview_array['pageview'] - pageview_array['pageview'].min()) / (pageview_array['pageview'].max() - pageview_array['pageview'].min())
            
    return dict(map(tuple, pagerank_array)), dict(map(tuple, pageview_array))


def calc_score_according_to_weight(pagerank_weight, pageviews_weight, title_res=[], title_weight=0, title_cos_sim={}, body_res=[], body_weight=0, body_cos_sim={}, anchor_res=[], anchor_weight=0, anchor_cos_sim={}):
    ''' Calculates results scores according to group (title, body, anchor, pagerank, pageviews) weights and returns a sorted list of 100 (wiki_id, title) pairs, descending by score.
        
     Arguments:
    --------
        title_res: list of (wiki_id: int, title: str)
            List of up to 100 search results, where each element is a tuple (wiki_id, title).
        
        title_weight: float
            A number between 0 to 1 which indicates the weight of the title results in the scoring formula.
            
        title_cos_sim: dict
            {wiki_id: cos_sim_score}
            
        pagerank_weight: float
            A number between 0 to 1 which indicates the weight of the pagerank results in the scoring formula.    
        
        pageviews_weight: float
            A number between 0 to 1 which indicates the weight of the pageview results in the scoring formula.    
        
        body_res: list of (wiki_id: int, title: str)
            List of up to 100 search results, where each element is a tuple (wiki_id, title).
            Default = [] for cases when we run this function for queries of len <= 2, we only want to use title_res, pagerank and pageview. 
            
        body_weight: float
            A number between 0 to 1 which indicates the weight of the body results in the scoring formula.
            
        body_cos_sim: dict
            {wiki_id: cos_sim_score}
            
        anchor_res: list of (wiki_id: int, title: str)
            List of up to 100 search results, where each element is a tuple (wiki_id, title).
            Default = [] for cases when we run this function for queries of len <= 2, we only want to use title_res, pagerank and pageview.
            
        anchor_weight: float
            A number between 0 to 1 which indicates the weight of the anchor results in the scoring formula.
            
        anchor_cos_sim: dict
            {wiki_id: cos_sim_score}
            
        
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    
    title_res_dict = dict(title_res)
    body_res_dict = dict(body_res)
    anchor_res_dict = dict(anchor_res)
    res_dict = dict(ChainMap(anchor_res_dict, title_res_dict, body_res_dict))  # merged in descending order regarding size to minimize number of merge confilcts

    pagerank_dict, pageviews_dict = normalize_result_set_pagerank_and_pageviews(res_dict)
    
    # Calculates a final score according to groups' weights, whether it appeared in the group's result dictionary & cosine similarity score
    return [wiki_id_title_score[0] for wiki_id_title_score in sorted({(wiki_id, title): title_weight * bool(title_res_dict.get(wiki_id, 0)) * title_cos_sim.get(wiki_id, 0) + body_weight * bool(body_res_dict.get(wiki_id, 0)) * body_cos_sim.get(wiki_id, 0) + anchor_weight * bool(anchor_res_dict.get(wiki_id, 0)) * anchor_cos_sim.get(wiki_id, 0) + pagerank_weight * pagerank_dict.get(wiki_id, 0) + pageviews_weight * pageviews_dict.get(wiki_id, 0) for wiki_id, title in res_dict.items()}.items(), key=lambda x: x[1], reverse=True)[:100]]
    

def weight_fine_tuning(search_query, tw, prw, pvw, bw ,aw):
    ''' Groups' weights fine-tuning for debugging purposes.
        
     Arguments:
    --------
        search_query: str or None
            Query String if in debug mode, None if query was sent via http
        
        tw: float
            A number between 0 to 1 which indicates the weight of the title results in the scoring formula.
            
        prw: float
            A number between 0 to 1 which indicates the weight of the pagerank results in the scoring formula.    
        
        pvw: float
            A number between 0 to 1 which indicates the weight of the pageview results in the scoring formula.    
        
        bw: float
            A number between 0 to 1 which indicates the weight of the body results in the scoring formula.

        aw: float
            A number between 0 to 1 which indicates the weight of the anchor results in the scoring formula.
            
    Returns:
    --------
        title_weight, pagerank_weight, pageviews_weight, body_weight, anchor_weight (floats)
    '''
    title_weight = float(request.args.get('title_weight', tw)) if not search_query else tw
    pagerank_weight = float(request.args.get('pagerank_weight', prw)) if not search_query else prw
    pageviews_weight = float(request.args.get('pageviews_weight', pvw)) if not search_query else pvw
    body_weight = float(request.args.get('body_weight', bw)) if not search_query else bw
    anchor_weight = float(request.args.get('anchor_weight', aw)) if not search_query else aw
    
    return title_weight, pagerank_weight, pageviews_weight, body_weight, anchor_weight
    
    
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
        
    Arguments:
    --------
        search_query: str or None
            Query String for when the query is not provided by an API call (debugging)
            
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    
    if search_query:
        query = search_query
        norm_cos_sim = False

    else:
        query = request.args.get('query', '')
        norm_cos_sim = request.args.get('norm_cos_sim', "False") in ["True", "true", "T", "t"] # Cosine Similarity Normalization switch for debugging purposes
        title_slice = int(request.args.get('title_slice', 5))
        body_slice = int(request.args.get('body_slice', 10))
    
    if len(query) == 0:
      return [] if search_query else jsonify([])
    
    # BEGIN SOLUTION
    
    tok_query = tokenize(query)
    
    # Longer queries mainly consider body scores, assisted by pagerank and pageviews (anchor had a negative influence on the M.A.P@40 score)
    # Queries that involve questions require explanations rather than retrieving an entity
    # Queries that contain a year are most likely event oriented and thus should go to a title based search and not body.
    if not bool(re.search(r'\b\d{4}\b', query)) and ((len(tok_query) >= 3) or (query[-1] == '?' and len(query.split()) >= 4) or (len(query.split()) >= 5)) :
        # weight fine-tuning for debugging purposes
        title_weight, pagerank_weight, pageviews_weight, body_weight, anchor_weight = weight_fine_tuning(search_query, 0, 0.1, 0.1, 0.8 ,0)
        
        #title_res, title_cos_sim = search_tfidf_cos_sim(query=query, group="title", norm_cos_sim=norm_cos_sim, is_search=True)
        body_res, body_cos_sim = search_tfidf_cos_sim(query=query, group="body", norm_cos_sim=norm_cos_sim, is_search=True)
        #anchor_res, anchor_cos_sim = search_tfidf_cos_sim(query=query, group="anchor", norm_cos_sim=norm_cos_sim, is_search=True)
         
        # While optimizing the weights that the anchor calculates only lowers our M.A.P@40 score.
        res = calc_score_according_to_weight(pagerank_weight=pagerank_weight, pageviews_weight=pageviews_weight, body_res=body_res, body_weight=body_weight, body_cos_sim=body_cos_sim)[:body_slice]#title_res=title_res, title_weight=title_weight, title_cos_sim=title_cos_sim, anchor_res=anchor_res, anchor_weight=anchor_weight, anchor_cos_sim=anchor_cos_sim) 
        
        
    else: # Wiki scores for short queries are calculated with a larger emphasis on the title, assisted by body, pagerank and pageviews (anchor had a negative influence on the M.A.P@40 score)
        # weight fine-tuning for debugging purposes
        title_weight, pagerank_weight, pageviews_weight, body_weight, anchor_weight = weight_fine_tuning(search_query, 0.8, 0.1, 0.1, 0 ,0)

        res, title_cos_sim = search_tfidf_cos_sim(query=query, group="title", norm_cos_sim=norm_cos_sim, is_search=True)
        
        res = calc_score_according_to_weight(title_res=res, title_weight=title_weight, title_cos_sim=title_cos_sim, pagerank_weight=pagerank_weight, pageviews_weight=pageviews_weight)[:title_slice]
        
    return res if search_query else jsonify(res)
    # END SOLUTION


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
        
     Arguments:
    --------
        search_query: str or None
            Query String for when the query is not provided by an API call (debugging / encapsulated search requested by search)
            
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    
    if search_query:
        query = search_query
        norm_cos_sim = False
    else:
        query = request.args.get('query', '')
        norm_cos_sim = request.args.get('norm_cos_sim', "False") in ["True", "true", "T", "t"] # Cosine Similarity Normalization switch for debugging purposes

    # BEGIN SOLUTION
    res = search_tfidf_cos_sim(query=query, group="body", norm_cos_sim=norm_cos_sim)
    return res if search_query else jsonify(res)
    # END SOLUTION


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
        
     Arguments:
    --------
        search_query: str or None
            Query String for when the query is not provided by an API call (debugging / encapsulated search requested by search)
            
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
    res = search_boolean(query=query, group="title")
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
    
     Arguments:
    --------
        search_query: str or None
            Query String for when the query is not provided by an API call (debugging / encapsulated search requested by search)
    
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
    res = search_boolean(query=query, group="anchor")
    return res if search_query else jsonify(res)
    # END SOLUTION


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank(wiki_ids_query=None):
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
        
     Arguments:
    --------
        wiki_ids_query: list or None
            List of wiki IDs for debugging
            
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    if wiki_ids_query:
        wiki_ids = wiki_ids_query
    else:
        wiki_ids = request.get_json()
    
    if len(wiki_ids) == 0:
      return [] if wiki_ids_query else jsonify([])
    
    # BEGIN SOLUTION
    filtered_ids = list(filter(lambda x: x in id_title_pr_pv_dict, wiki_ids))

    res = [(id_title_pr_pv_dict[id][0], id_title_pr_pv_dict[id][1]) for id in filtered_ids]

    return res if wiki_ids_query else jsonify(res)
    # END SOLUTION
    

@app.route("/get_pageview", methods=['POST'])
def get_pageview(wiki_ids_query=None):
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
        
    Arguments:
    --------
        wiki_ids_query: list or None
            List of wiki IDs for debugging
            
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    
    if wiki_ids_query:
        wiki_ids = wiki_ids_query
    else:
        wiki_ids = request.get_json()
    
    if len(wiki_ids) == 0:
      return [] if wiki_ids_query else jsonify([])
    
    # BEGIN SOLUTION
    
    filtered_ids = list(filter(lambda x: x in id_title_pr_pv_dict, wiki_ids))

    res = [(id_title_pr_pv_dict[id][0], id_title_pr_pv_dict[id][2]) for id in filtered_ids]

    return res if wiki_ids_query else jsonify(res)
    # END SOLUTION


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


def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., wiki_id)
                                                                value: similarity score. We keep up to 5 digits after the                                                                  decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (wiki_id, score) in the length of N.
    """

    return sorted([(wiki_id, round(score, 10)) for wiki_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


def load_pickle_from_bucket(path):
    """
    Downloads a referenced .pkl file as bytes from the bucket and loads it as a variable. used in load_data(), at engine startup

    Parameters:
    -----------
    path: str
        Path of referenced .pkl file within the bucket

    Returns:
    -----------
    Object stored as a .pkl file within the bucket 
    """
    
    client = storage.Client()
    bucket = client.bucket('318419512_318510252')
    file = pickle.loads(bucket.get_blob(path).download_as_bytes())
    client.close()
    gc.collect()
    
    return file


@app.route("/load_data")
def load_data():
    """
    Loads all necessary variables from the GCP Bucket as Global Variables. Runs before the server goes up.
    
    Parameters:
    -----------
    None
    
    Returns:
    -----------
    None
    """
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Prevents kernel death when downloading .pkl files from bucket
    bucket_name = '318419512_318510252'
    id_title_pr_pv_dict_path = 'gs://318419512_318510252/id_title_pr_pv_dict_path.pkl'
    # load the three indexes
    global idx_body
    global idx_title
    global idx_anchor
    idx_body = load_pickle_from_bucket("postings_gcp/body_index.pkl")
    idx_title = load_pickle_from_bucket("postings_gcp/title_index.pkl")
    idx_anchor = load_pickle_from_bucket("postings_gcp/anchor_index.pkl")

    # declare the variables we will need in other functions (search) as global
    global id_pagerank_title_pageviews_df
    global id_title_pr_pv_dict
    global N  # Number of documents in the corpus
    
    N = len(idx_body.DL)

    id_title_pr_pv_dict = load_pickle_from_bucket("id_title_pr_pv_dict_cast.pkl")

    # as the .pkl files are loaded with a lazy manner, we explicitly initiate their mapping to dict in load_data() by a dummy call
    idx_body.DL[309]
    idx_title.DL[309]
    idx_anchor.DL[309]
    id_title_pr_pv_dict[309]  
    

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    load_data()
    app.run(host='0.0.0.0', port=8080, debug=True)
    

    
    
