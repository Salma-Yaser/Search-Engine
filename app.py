#python
import os
import streamlit as st
import nltk

# Try to download NLTK resources first - this is needed for deployment
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except Exception as e:
    st.warning(f"Error downloading NLTK resources: {str(e)}")

# Now import modules that depend on NLTK resources
try:
    from preprocess import preprocess
    from indexing import build_vocabulary, build_term_doc_matrix, build_inverted_index
    from retrival import (
        search_term_doc_incidence,
        search_inverted_index,
        search_tfidf,
    )
except Exception as e:
    st.error(f"Error importing modules: {str(e)}")
    st.stop()

# -- Load and cache data and indexes -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_index(data_dir="Dataset"):
    # Read all .txt files
    filenames = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])
    raw_docs = []
    for f in filenames:
        with open(os.path.join(data_dir, f), encoding="utf-8") as file:
            raw_docs.append(file.read())

    # Preprocess documents
    pre_docs = [preprocess(doc) for doc in raw_docs]

    # Build vocabulary and indexes
    vocabulary = build_vocabulary(pre_docs) #returns a sorted list of all unique terms across documents.
    term_doc_matrix = build_term_doc_matrix(pre_docs, vocabulary)
    inverted_index = build_inverted_index(pre_docs)

    # For TF-IDF, join tokens back to string form
    tfidf_docs = [" ".join(tokens) for tokens in pre_docs]
    return filenames, raw_docs, pre_docs, vocabulary, term_doc_matrix, inverted_index, tfidf_docs

filenames, raw_docs, pre_docs, vocabulary, term_doc_matrix, inverted_index, tfidf_docs = load_and_index()

# -- Helper for snippet display ------------------------------------------------------
def get_snippet(text, terms, radius=50):
    """
    Return a snippet of text around the first occurrence of any term in terms.
    """
    lower = text.lower()
    for term in terms:
        idx = lower.find(term.lower())
        if idx != -1:
            start = max(0, idx - radius)
            end = min(len(text), idx + radius)
            return text[start:end].strip().replace("\n", " ")
    # fallback: start of document
    return text[:radius*2].strip().replace("\n", " ")

# -- Streamlit UI ---------------------------------------------------------------------
# Title and description
st.markdown(
    """
    <h1 style='
        text-align: center;
        font-size: 3em;
        color: #6f91b5;
        font-family: "Habibi", sans-serif;
        margin-top: 0.5em;
        margin-bottom: 0.5em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    '>
         Search Engine
    </h1>
    """,
    unsafe_allow_html=True
)

# background image 
# Create a container div for the search background
st.markdown(
    """
    <style>
    .search-container {
        background-image: url("https://ik.imagekit.io/4f4cbqxyi/photo-1651352076676-58a34812f3d2.avif");
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    
    <div class="search-container">
        <div style="height: 60px;"></div>
    </div>
    """,
    unsafe_allow_html=True
)

# User inputs
def main():
    #user inputs    
    query = st.text_input("", placeholder="Enter your search query here...")
    #-----------------------------------------------
    model = st.radio(
        "Choose search method:",
        ("Term-Document Incidence", "Inverted Index", "TF-IDF with Cosine Similarity"),
    )
    if model == "TF-IDF with Cosine Similarity":
        k = st.slider("Number of results (top k):", min_value=1, max_value=20, value=5)
    else:
        k = None

    if st.button("Search"):
        if not query:
            st.warning("Please enter a query to search.")
            return

        # Preprocess query
        query_terms = preprocess(query)
        st.write("🔍 Processed query terms:", query_terms)

        # Perform search
        if model == "Term-Document Incidence":
            hits = search_term_doc_incidence(query_terms, term_doc_matrix, vocabulary)
            st.write(f"✅ Found {len(hits)} documents containing all query terms.")
            for i in hits:
                st.subheader(filenames[i])
                st.write(get_snippet(raw_docs[i], query_terms))

        elif model == "Inverted Index":
            hits = search_inverted_index(query_terms, inverted_index)
            st.write(f"✅ Found {len(hits)} documents containing all query terms.")
            for i in hits:
                st.subheader(filenames[i])
                st.write(get_snippet(raw_docs[i], query_terms))

        else:
            results = search_tfidf(tfidf_docs, query, top_k=k)
            st.write(f"📈 Top {k} documents by cosine similarity:")
            for i, score in results:
                st.subheader(f"{filenames[i]} (score: {score:.3f})")
                st.write(get_snippet(raw_docs[i], query_terms))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please check if NLTK data is properly downloaded and available.")

# -- End of Streamlit app -----------------------------------------------------------