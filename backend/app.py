import os
import json
import re
import math
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import consts

app = Flask(__name__)
CORS(app)

feedback_data = {
    "relevant" : set(),
    "irrelevant" : set()
}

# Load data from JSON
def load_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data).T

# lowercase, remove punctuation, split into words
def preprocess(text):
    return re.sub(r'[^\w\s]', '', text.lower()).split()

# Calculate IDF values based on doc
def calculate_idf(corpus):
    df_dict = {}
    n_documents = len(corpus)
    for document in corpus:
        words = set(preprocess(document))
        for word in words:
            df_dict[word] = df_dict.get(word, 0) + 1
    return {word: math.log(n_documents / (df + 1)) + 1 for word, df in df_dict.items()}

# Compute TF-IDF vector for texts
def compute_tfidf_vector(text, idf_dict):
    words = preprocess(text)
    tf_dict = {word: words.count(word) for word in words if word in idf_dict}
    tfidf_vector = np.zeros(len(idf_dict))
    for word, tf in tf_dict.items():
        index = list(idf_dict.keys()).index(word)
        tfidf_vector[index] = tf * idf_dict[word]
    return tfidf_vector

# Create TF-IDF matrix for all documents
def create_tfidf_matrix(pitches_df, idf_dict):
    return np.array([compute_tfidf_vector(doc, idf_dict) for doc in pitches_df['Pitched_Business_Desc']])

current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'helpers/superclean_pitchesdeals.json')
pitches_df = load_data(json_file_path)
corpus = pitches_df['Pitched_Business_Desc'].tolist()
idf_dict = calculate_idf(corpus)
tfidf_matrix = create_tfidf_matrix(pitches_df, idf_dict)
U, S, Vt = np.linalg.svd(tfidf_matrix, full_matrices=False)

# Flask routes
@app.route("/")
def home():
    return render_template('base.html', title="Document Search with SVD")

@app.route("/pitches")
def pitches_search():
    query = request.args.get("query")
    if not query:
        return {"error": "No query provided"}, 400
    result = svd_search(query, U, S, Vt, pitches_df, idf_dict)
    return result or {"message": "No matches found"}, 404


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()  # Retrieve JSON data from the request
    feedback_type = data.get('feedback')  # Get the feedback type from the data
    doc_id = data.get('doc_id')

    if feedback_type == "relevant":
        feedback_data["relevant"].add(doc_id)
        feedback_data["irrelevant"].discard(doc_id)
    elif feedback_type == "irrelevant":
        feedback_data["irrelevant"].add(doc_id)
        feedback_data["relevant"].discard(doc_id)

    print(f"Received feedback: {feedback_type} for document ID: {doc_id}")
    print(f"Relevant Docs: {feedback_data['relevant']}")
    print(f"Irrelevant Docs: {feedback_data['irrelevant']}")

    return jsonify(success=True)

# def compute_query_vector(query, idf_dict):
#     """
#     Compute the TF-IDF vector for the query based on the provided IDF values.
#     """
#     query_terms = query.lower().split()
#     tf_query = {term: query_terms.count(term) for term in set(query_terms)}

#     tfidf_query = np.zeros(len(idf_dict))
#     for term, tf in tf_query.items():
#         if term in idf_dict:
#             index = list(idf_dict.keys()).index(term)
#             tfidf_query[index] = tf * idf_dict[term]
#     return tfidf_query

# def compute_document_vector(doc_id, inverted_index, idf_dict, n_docs):
#     tfidf_vector = np.zeros(len(idf_dict))
#     for term, idf_value in idf_dict.items():
#         if term in inverted_index:
#             doc_occurrences = [freq for doc, freq in inverted_index[term] if doc == doc_id]
#             if doc_occurrences:
#                 tf = doc_occurrences[0]
#                 tfidf_vector[list(idf_dict.keys()).index(term)] = tf * idf_value
#     return tfidf_vector

# def cosine_similarity(tfidf_doc, tfidf_query, doc_norm):
#     """
#     Calculate the cosine similarity between a document and the query.
#     """
#     if np.linalg.norm(tfidf_query) == 0 or doc_norm == 0:
#         return 0
#     return np.dot(tfidf_doc, tfidf_query) / (doc_norm * np.linalg.norm(tfidf_query))

# def json_search(query):

#     query_words = set(query.lower().split())
#     query_word_count = len(query_words)
#     tfidf_query = compute_query_vector(query, consts.idf_dict)
#     similarities = []

#     print("this is iterrows:")
#     print(pitches_df.iterrows())
#     print("")

#     for index, row in pitches_df.iterrows():
#         print("this is index: "+ index)
#         print("this is row: "+ row)
#         tfidf_doc = compute_document_vector(index, consts.inverted_index, consts.idf_dict, consts.n_docs)
#         doc_norm = consts.doc_norms[int(row)]
#         similarity = cosine_similarity(tfidf_doc, tfidf_query, doc_norm)
#         similarities.append((index, similarity))


def create_tfidf_matrix(document_list):
    """ 
    Create the TF-IDF matrix. 
    """
    tfidf_matrix = np.array([compute_document_vector(index, consts.inverted_index, consts.idf_dict, consts.n_docs) for index, row in document_list])
    return tfidf_matrix


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

def compute_query_vector(query, idf_dict):
    """
    Compute the TF-IDF vector for the query based on the provided IDF values.
    """
    words = preprocess(query)
    tf_query = {word: words.count(word) for word in set(words) if word in idf_dict}

    tfidf_query = np.zeros(len(idf_dict))
    for word, tf in tf_query.items():
        if word in idf_dict:
            index = list(idf_dict.keys()).index(word)
            tfidf_query[index] = tf * idf_dict[word]
    return tfidf_query

def json_search(query):
    tfidf_query = compute_query_vector(query, idf_dict)
    if np.linalg.norm(tfidf_query) == 0:
        return []  # Return an empty list if the query vector is zero

    similarities = []
    for index, row in pitches_df.iterrows():
        # Compute the TF-IDF vector for each document
        tfidf_doc = compute_tfidf_vector(row['Pitched_Business_Desc'], idf_dict)
        doc_norm = np.linalg.norm(tfidf_doc)
        
        # Compute cosine similarity
        if doc_norm != 0 and np.linalg.norm(tfidf_query) != 0:
            similarity = np.dot(tfidf_doc, tfidf_query) / (doc_norm * np.linalg.norm(tfidf_query))
        else:
            similarity = 0
        similarities.append(similarity)

    return similarities

def rocchio(query, a=.3, b=.3, c=.8, clip = True):
    """Returns a vector representing the modified query vector.

    Note:
        If the `clip` parameter is set to True, the resulting vector should have
        no negatve weights in it!
    """
    q0 = query
    relevant_docs = list(feedback_data["relevant"])
    irrelevant_docs = list(feedback_data["irrelevant"])

    rel_docs_indices = [consts.mapping[doc] for doc in relevant_docs]
    irrel_docs_indices = [consts.mapping[doc] for doc in irrelevant_docs]

    term1 = a * q0

    if len(relevant_docs) == 0:
        term2 = b * np.zeros_like(q0)
    else:
        term2 = b * np.mean(tfidf_matrix[rel_docs_indices], axis=0)

    if len(irrelevant_docs) == 0:
        term3 = c * np.zeros_like(q0)
    else:
        term3 = c * np.mean(tfidf_matrix[irrel_docs_indices], axis=0)

    res = term1 + term2 - term3

    if clip:
        for i in range(len(res)):
            if res[i] < 0:
                res[i] = 0

    return res

def svd_search(query, U, S, Vt, pitches_df, idf_dict):
    cos_similarity = json_search(query)
    query_vector = compute_tfidf_vector(query, idf_dict)
    updated_query_vector = rocchio(query_vector)
    print(f"query vec: {query_vector}")
    print(f"rocchio query vec: {updated_query_vector}")


    if np.linalg.norm(query_vector) == 0:
        print("Query vector is zero.")
        return None
    query_svd = updated_query_vector @ Vt.T
    query_svd /= np.linalg.norm(query_svd)
    document_projections = U * S[:len(query_svd)]
    similarities = np.dot(document_projections, query_svd)


    # print("len of soc sims: " + repr(len(social_similarities)))

    # print("")
    # print("len of svd sims" + repr(len(similarities)))
    # print("len of cos sims" + repr(len(cos_similarity)))
    # print("")
    # print("svd sims")
    # print(similarities)
    # print("")
    # print("cos sims")
    # print(cos_similarity)
    # print("")

    # agg_sims = similarities
    cos_similarity = np.array(cos_similarity)
    similarities = np.array(similarities)

    # Normalize cos similarities
    if np.any(cos_similarity):
        cos_similarity = (cos_similarity - np.min(cos_similarity)) / (np.max(cos_similarity) - np.min(cos_similarity))

    # Normalize svd similarities
    if np.any(similarities):
        similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))

    # social comp
    max_viewership = pitches_df['US_Viewership'].max()
    min_viewership = pitches_df['US_Viewership'].min()
    pitches_df['Normalized_Viewership'] = (pitches_df['US_Viewership'] - min_viewership) / (max_viewership - min_viewership)
    social_similarities = similarities * (pitches_df['Normalized_Viewership'].values) # EDIT SOCIAL COMP EFFECT

    # print("new svd sims")
    # print(social_similarities)
    # print("")
    # print("new cos sims")
    # print(cos_similarity)
    # print("")

    agg_sims = ((0.9 * cos_similarity) + (0.1 * social_similarities))
    # agg_sims = cos_similarity
    # agg_sims = similarities

    # top_indices = np.where(agg_sims > 0.5)[0]
    # if len(top_indices) == 0:
    #     return None
    agg_sims = np.nan_to_num(agg_sims)  # Convert NaN to 0
    top_indices = np.where(agg_sims > 0.5)[0]
    
    # matches_filtered = pitches_df.iloc[top_indices]
    # matches_filtered["similarity_score"] = agg_sims[top_indices]
    matches_filtered = pitches_df.iloc[top_indices].copy()  # Ensure it's a copy
    matches_filtered.loc[:, "similarity_score"] = agg_sims[top_indices]

    matches_filtered = matches_filtered.sort_values(by="similarity_score", ascending=False)

    return matches_filtered.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)