import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
import math
import consts

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

# trying to preprocess

# end of preprocess
app = Flask(__name__)
CORS(app)


def json_search(query):

    query_words = set(query.lower().split())
    query_word_count = len(query_words)
    ### WE SHOULD REMOVE STOP WORDS ###
    tfidf_query = compute_query_vector(query, consts.idf_dict) 
    similarities = []
    # This is a numpy array
    tfidf_matrix = create_tfidf_matrix(pitches_df.iterrows())
    # SVD
    U, sigma, Vt = np.linalg.svd(tfidf_matrix, full_matrices=False)
    # Choose the top k components, can do different k
    k = 35
    V_k = Vt.T[:, :k]
    normalize_vec = V_k/np.linalg.norm(V_k, axis = 1, keepdims = True)
    # putting the query in k dimensions like V_k
    tfidf_query = np.dot(V_k.T, tfidf_query[:V_k.shape[0]])
    normal_query = tfidf_query / np.linalg.norm(tfidf_query)
    similarities = np.dot(normalize_vec, normal_query)

    # Old cosine stuff
    # for index, row in pitches_df.iterrows():
    #     tfidf_doc = compute_document_vector(index, consts.inverted_index, consts.idf_dict, consts.n_docs)
    #     doc_norm = consts.doc_norms[index]
    #     similarity = cosine_similarity(tfidf_doc, tfidf_query, doc_norm)
    #     similarities.append((index, similarity))
    
    # Social component starts here
    # We should probably make it so a high viewership weights high similarities higher and low similarities 
    # lower because particularly good or bad deals likely get more viewership
    # Right now we just weight similarity as 10% of the score 
    max_viewership = pitches_df['US_Viewership'].max()
    min_viewership = pitches_df['US_Viewership'].min()
    pitches_df['Normalized_Viewership'] = (pitches_df['US_Viewership'] - min_viewership) / (max_viewership - min_viewership)
    social_similarities = [sim +  (.1 * pitches_df.iloc[ind]['Normalized_Viewership']) for ind, sim in enumerate(similarities)]
    # End of social component

    indexed_similarities = [(index, sim) for index, sim in enumerate(social_similarities)]
    # Sort documents based on similarity
    indexed_similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = [ind for ind, sim in indexed_similarities if sim > 0.5]  # Could change threshold to be greater or less than 0.5

    matches_filtered = pitches_df.iloc[top_matches]
    matches_filtered = matches_filtered[['Pitched_Business_Identifier', 'Pitched_Business_Desc', 'Deal_Status', 'Deal_Shark', 'US_Viewership']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json


    # matches_filtered = matches[['Pitched_Business_Identifier', 'Pitched_Business_Desc', 'Deal_Status', 'Deal_Shark', 'US_Viewership']]
    # matches_filtered_json = matches_filtered.to_json(orient='records')
    # return matches_filtered_json


def create_tfidf_matrix(document_list):
    """ 
    Create the TF-IDF matrix. 
    """
    tfidf_matrix = np.array([compute_document_vector(index, consts.inverted_index, consts.idf_dict, consts.n_docs) for index, row in document_list])
    return tfidf_matrix

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
