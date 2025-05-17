import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def search_term_doc_incidence(query_terms, term_doc_matrix, vocabulary):
    """
    Document-Term Incidence: return doc indices where ALL query_terms appear.

    Args:
        query_terms (list of str): preprocessed query tokens.
        term_doc_matrix (np.ndarray): binary matrix shape (|vocab|, |docs|).
        vocabulary (list of str): list of terms matching matrix rows.

    Returns:
        List[int]: sorted list of document indices containing all terms.
    """
    # map terms to row indices
    term_to_idx = {t: i for i, t in enumerate(vocabulary)}
    # find rows for query terms
    rows = [term_doc_matrix[term_to_idx[t]] for t in query_terms if t in term_to_idx]
    if not rows:
        return []
    # intersect document columns where (all) rows == 1
    docs = np.all(np.vstack(rows) == 1, axis=0)
    return list(np.nonzero(docs)[0]) #This gets the indices of documents where docs == True


def search_inverted_index(query_terms, inverted_index):
    """
    Inverted Index: return doc indices containing ALL query terms.

    Args:
        query_terms (list of str): preprocessed query tokens.
        inverted_index (dict): term -> list of doc IDs.

    Returns:
        List[int]: sorted list of document indices containing all terms.
    """
    # get posting lists
    postings = [set(inverted_index.get(t, [])) for t in query_terms]
    if not postings:
        return []
    # intersect all sets
    result = set.intersection(*postings)
    return sorted(result)


def search_tfidf(raw_documents, query, top_k=6):
    """
    TF-IDF with cosine similarity: rank docs by similarity to query.

    Args:
        raw_documents (list of str): original document texts.
        query (str): raw query string.
        top_k (int): number of top results to return.

    Returns:
        List[tuple[int, float]]: list of (doc_idx, score) sorted by score desc.
    """
    # vectorize documents and query
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(raw_documents)
    query_vec = vectorizer.transform([query])
    # compute cosine similarities
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]


