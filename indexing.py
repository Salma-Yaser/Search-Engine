import numpy as np
from collections import defaultdict


def build_vocabulary(preprocessed_docs):
    """
    Build a sorted list of unique terms (vocabulary) from preprocessed documents.

    Args:
        preprocessed_docs (list of list of str): Tokenized documents.

    Returns:
        List[str]: Sorted vocabulary.
    """
    vocab = set()
    for doc in preprocessed_docs:
        vocab.update(doc)
    return sorted(vocab)


def build_term_doc_matrix(preprocessed_docs, vocabulary):
    """
    Build the term-document incidence matrix (binary).

    Args:
        preprocessed_docs (list of list of str): Tokenized documents.
        vocabulary (list of str): List of all terms.

    Returns:
        np.ndarray: 2D binary matrix of shape (|vocabulary|, |documents|).
    """
    term_to_index = {term: i for i, term in enumerate(vocabulary)}
    matrix = np.zeros((len(vocabulary), len(preprocessed_docs)), dtype=int)
    for doc_idx, tokens in enumerate(preprocessed_docs):
        for term in set(tokens):
            if term in term_to_index:
                matrix[term_to_index[term], doc_idx] = 1
    return matrix


def build_inverted_index(preprocessed_docs):
    """
    Build an inverted index mapping terms to lists of document indices.

    Args:
        preprocessed_docs (list of list of str): Tokenized documents.

    Returns:
        dict[str, list[int]]: Inverted index.
    """
    index = defaultdict(list)
    for doc_idx, tokens in enumerate(preprocessed_docs):
        for term in set(tokens):
            index[term].append(doc_idx)
    return dict(index)
