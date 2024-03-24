import json
import os
import pandas as pd
import numpy as np
import math
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'helpers/pitchesdeals.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    pitches_df = pd.DataFrame.from_dict(data, orient='index').reset_index(drop=True)

# trying to preprocess
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

def compute_idf(inv_idx, n_docs, min_df=5, max_df_ratio=0.95):
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

descr = pitches_df['Pitched_Business_Desc']
descriptions = [None] * len(descr)
n_docs = len(descriptions)
for ind in range(n_docs):
    toks = descr[ind].lower().split()
    descriptions[ind] = list(set(toks))
inverted_index = build_inverted_index(descriptions)
idf_dict = compute_idf(inverted_index, n_docs)
doc_norms = compute_doc_norms(inverted_index, idf_dict, n_docs)