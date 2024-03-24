import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
import math

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'helpers/pitchesdeals.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    pitches_df = pd.DataFrame.from_dict(data, orient='index').reset_index(drop=True)

app = Flask(__name__)
CORS(app)


def json_search(query):

    query_words = set(query.lower().split())
    query_word_count = len(query_words)
    # check if any query word is in the pitch description
    descr = pitches_df['Pitched_Business_Desc']
    descriptions = [None] * len(descr)  # Use a Python list
    n_docs = len(descriptions)
    for ind in range(n_docs):
        toks = descr[ind].lower().split()
        descriptions[ind] = list(set(toks))
    inverted_index = build_inverted_index(descriptions)
    idf_dict = compute_idf(inverted_index, n_docs)
    doc_norms = compute_doc_norms(inverted_index, idf_dict, n_docs)
    tfidf_query = compute_query_vector(query, idf_dict)
    similarities = []
    for index, row in pitches_df.iterrows():
        tfidf_doc = compute_document_vector(index, inverted_index, idf_dict, n_docs)
        doc_norm = doc_norms[index]
        similarity = cosine_similarity(tfidf_doc, tfidf_query, doc_norm)
        similarities.append((index, similarity))
    
    # Sort documents based on similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = list(set([sim for sim in similarities if sim[1] > 0.3]))  # Could change threshold to be greater or less than 0.5
    matches_filtered = pitches_df.iloc[[sim[0] for sim in top_matches]]
    matches_filtered = matches_filtered[['Pitched_Business_Identifier', 'Pitched_Business_Desc', 'Deal_Status', 'Deal_Shark', 'US_Viewership']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json


    # matches_filtered = matches[['Pitched_Business_Identifier', 'Pitched_Business_Desc', 'Deal_Status', 'Deal_Shark', 'US_Viewership']]
    # matches_filtered_json = matches_filtered.to_json(orient='records')
    # return matches_filtered_json


def build_inverted_index(pitches):
    """Builds an inverted index from the messages. This is taken and modified from a4. 

    Arguments
    =========

    pitches: list of list of strings.
        Each entry in this list is a list of all the tokens in the pitch.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    """
    inverted_index = {}
    for i in range(len(pitches)):
        pitch = pitches[i]
        freqs = {}
        for tok in pitch:
            if tok in freqs:
                freqs[tok] += 1
            else:
                freqs[tok] = 1
        for tok in freqs: 
            ele = freqs[tok]
            if tok in inverted_index: 
                inverted_index[tok].append((i, ele))
            else:
                inverted_index[tok] = [(i, ele)]
    for tok in inverted_index:
        inverted_index[tok].sort(key=lambda x: x[0])
    return inverted_index

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index. This was taken directly from a4. 
    Words that are too frequent or too infrequent get pruned.

    Hint: Make sure to use log base 2.

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """

    # TODO-5.1
    idf_dict = {}
    max_doc = max_df_ratio * n_docs
    for word in inv_idx:
        pair_list = inv_idx[word]
        size = len(pair_list)
        if min_df <= size and size <= max_doc:
            idf_dict[word] = math.log2(n_docs/(1 + size))
        
    return idf_dict

def compute_doc_norms(index, idf, n_docs):
    """Precompute the euclidean norm of each document. Taken entirely from a4. 
    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.
    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """

    # TODO-6.1
    norms = np.zeros(n_docs)
    for word in index:
        pair_list = index[word]
        if word in idf:
            word_idf = idf[word]
            for doc_id, count in pair_list:
                norms[doc_id] += (count * word_idf) ** 2
    norms = np.sqrt(norms)
    return norms

def compute_query_vector(query, idf_dict):
    """
    Compute the TF-IDF vector for the query based on the provided IDF values.
    """
    query_terms = query.lower().split()
    tf_query = {term: query_terms.count(term) for term in set(query_terms)}

    tfidf_query = np.zeros(len(idf_dict))
    for term, tf in tf_query.items():
        if term in idf_dict:
            index = list(idf_dict.keys()).index(term)
            tfidf_query[index] = tf * idf_dict[term]
    return tfidf_query

def compute_document_vector(doc_id, inverted_index, idf_dict, n_docs):
    tfidf_vector = np.zeros(len(idf_dict))
    for term, idf_value in idf_dict.items():
        if term in inverted_index:
            doc_occurrences = [freq for doc, freq in inverted_index[term] if doc == doc_id]
            if doc_occurrences:
                tf = doc_occurrences[0]
                tfidf_vector[list(idf_dict.keys()).index(term)] = tf * idf_value
    return tfidf_vector

def cosine_similarity(tfidf_doc, tfidf_query, doc_norm):
    """
    Calculate the cosine similarity between a document and the query.
    """
    if np.linalg.norm(tfidf_query) == 0 or doc_norm == 0:
        return 0
    return np.dot(tfidf_doc, tfidf_query) / (doc_norm * np.linalg.norm(tfidf_query))
    
@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")

@app.route("/pitches")
def pitches_search():
    # Get the user input from query parameters
    text = request.args.get("query")

    # Check if the text parameter is provided
    if not text:
        # Return an error message or an empty JSON if no query parameter is provided
        return {"error": "No query provided"}, 400  # 400 Bad Request

    # Call the search function with the user input
    result = json_search(text)

    # Check if the search result is empty
    if not result:
        # Return a message indicating no matches found
        return {"message": "No matches found"}, 404  # 404 Not Found

    return result


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
