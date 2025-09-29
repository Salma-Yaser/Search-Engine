import os
import nltk
import base64
import logging
from pathlib import Path
nltk.download('punkt')
import streamlit as st
from preprocess import preprocess
from indexing import build_vocabulary, build_term_doc_matrix, build_inverted_index
from retrival import (
    search_term_doc_incidence,
    search_inverted_index,
    search_tfidf,
)

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
GIF_URL = "https://ik.imagekit.io/tosp1g2et/img3.jpeg?updatedAt=1747670323821"

# --- Helper Functions ---
def get_snippet(text, terms, radius=50):
    """Returns all sentences containing search terms with highlighting"""
    snippets = []
    sentences = nltk.sent_tokenize(text)
    
    for sentence in sentences:
        lower_sentence = sentence.lower()
        found = any(term.lower() in lower_sentence for term in terms)
        if found:
            # Highlight all terms in the sentence
            highlighted = sentence
            for term in terms:
                highlighted = highlighted.replace(term, f'<span class="highlight">{term}</span>')
                highlighted = highlighted.replace(term.title(), f'<span class="highlight">{term.title()}</span>')
                highlighted = highlighted.replace(term.upper(), f'<span class="highlight">{term.upper()}</span>')
            snippets.append(highlighted)
    
    if snippets:
        # Join snippets with ellipsis and limit total length
        combined = " [...] ".join(snippets)
        if len(combined) > 1000:  # Prevent very long results
            combined = combined[:1000] + " [...]"
        return combined
    else:
        # If no sentences found, return beginning of text
        beginning = text[:radius*2].strip().replace("\n", " ")
        return f"{beginning} [...]"

# --- Data Loading ---
@st.cache_data(show_spinner=False)
def load_and_index(data_dir="Dataset"):
    try:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset folder not found at: {data_dir}")
            
        filenames = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])
        if not filenames:
            raise ValueError("No .txt files found in Dataset folder")

        raw_docs = []
        for f in filenames:
            file_path = os.path.join(data_dir, f)
            with open(file_path, encoding="utf-8") as file:
                content = file.read()
                if not content.strip():
                    logging.warning(f"Empty file: {f}")
                raw_docs.append(content)

        pre_docs = [preprocess(doc) for doc in raw_docs]
        vocabulary = build_vocabulary(pre_docs)
        term_doc_matrix = build_term_doc_matrix(pre_docs, vocabulary)
        inverted_index = build_inverted_index(pre_docs)
        tfidf_docs = [" ".join(tokens) for tokens in pre_docs]
        
        logging.info(f"Successfully loaded {len(filenames)} documents")
        return {
            "filenames": filenames,
            "raw_docs": raw_docs,
            "pre_docs": pre_docs,
            "vocabulary": vocabulary,
            "term_doc_matrix": term_doc_matrix,
            "inverted_index": inverted_index,
            "tfidf_docs": tfidf_docs
        }
    
    except Exception as e:
        logging.exception("Data loading failed")
        return None

# --- Main App ---
def main():
    st.set_page_config(
        page_title="Doc Discover",
        page_icon="🔍",
        layout="centered"
    )
    
    # Custom CSS with all requested styling
    st.markdown(f"""
        <style>
            /* Main background */
            .stApp {{
                background: url('{GIF_URL}') no-repeat center center fixed;
                background-size: cover;
            }}
            
            /* Main container */
            .main-container {{
                background-color: transparent;
                padding: 1rem;
                margin: 1rem auto;
                max-width: 900px;
            }}
            
            /* Search container - Transparent with curved corners */
            .search-container {{
                background-color: rgba(0, 0, 0, 0.5);
                border-radius: 20px;
                padding: 2rem;
                margin-bottom: 1rem;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s ease;
            }}
            
            .search-container:hover {{
                background-color: rgba(0, 0, 0, 0.6);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            
            /* Search input field - Light color with curved corners */
            .stTextInput>div>div>input {{
                background-color: rgba(0, 0, 0, 0.3) !important;
                backdrop-filter: blur(10px);
                color:white !important;
                border-radius: 12px !important;
                padding: 0.75rem 1rem !important;
                transition: all 0.3s ease;
                border: none !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            .stTextInput>div>div>input:focus {{
                box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.5);
                outline: none !important;
            }}
            
            /* Placeholder text color */
            .stTextInput>div>div>input::placeholder {{
                color: #666 !important;
            }}
            
            /* Search button - Curved with hover effect */
            .search-btn {{
                width: 100%;
                margin-top: 1rem;
                background-color: rgba(255, 77, 77, 0.8);
                color: white;
                border: none;
                padding: 0.75rem 1rem;
                border-radius: 12px;
                font-weight: bold;
                font-size: 1.1rem;
                transition: all 0.3s ease;
                backdrop-filter: blur(5px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            
            .search-btn:hover {{
                background-color: rgba(255, 77, 77, 0.9);
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(255, 77, 77, 0.3);
            }}
            
            /* Results box - Dark with curved corners */
            .result-box {{
                padding: 1.5rem;
                margin: 1rem 0;
                border-radius: 15px;
                background-color: rgba(30, 30, 30, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
                color: #f0f0f0;
                line-height: 1.6;
                backdrop-filter: blur(5px);
                transition: all 0.3s ease;
            }}
            
            .result-box:hover {{
                transform: translateY(-3px);
                box-shadow: 0 12px 24px rgba(0,0,0,0.3);
            }}
            
            /* Highlight for search terms */
            .highlight {{
                background-color: #ffcc00;
                color: #000;
                padding: 0 4px;
                border-radius: 4px;
                font-weight: bold;
            }}
            
            /* Headers */
            h1, h2, h3 {{
                color: #fff;
                text-shadow: 0 2px 4px rgba(0,0,0,0.5);
            }}
            
            /* Radio buttons - Transparent with curved design */
            .stRadio > div {{
                flex-direction: row;
                gap: 1rem;
                background-color: transparent !important;
            }}
            
            .stRadio > div > label {{
                color: white !important;
                background-color: rgba(255, 255, 255, 0.1) !important;
                padding: 0.75rem 1.5rem !important;
                border-radius: 12px !important;
                transition: all 0.3s ease;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin: 0.25rem 0 !important;
            }}
            
            .stRadio > div > label:hover {{
                background-color: rgba(255, 255, 255, 0.2) !important;
                transform: translateY(-2px);
            }}
            
            [data-baseweb="radio"] div:first-child {{
                border-color: rgba(255, 255, 255, 0.5) !important;
            }}
            
            /* Slider styling - Curved design */
              .stRadio > div {{
                flex-direction: row;
                gap: 1rem;
                background-color: transparent !important;
            }}
            
            .stRadio > div > label {{
                color: white !important;
                background-color: rgba(255, 255, 255, 0.1) !important;
                padding: 0.75rem 1.5rem !important;
                border-radius: 12px !important;
                transition: all 0.3s ease;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin: 0.25rem 0 !important;
            }}
            
            .stRadio > div > label:hover {{
                background-color: rgba(255, 255, 255, 0.2) !important;
                transform: translateY(-2px);
            }}
            
            [data-baseweb="radio"] div:first-child {{
                border-color: rgba(255, 255, 255, 0.5) !important;
            }}
            
            /* Slider styling - Curved design */


            
            /* تنسيق Radio Buttons - تأثير زجاجي */
        .stRadio > div {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .stRadio > div > label {{
            color: white !important;
            padding: 0.5rem 1rem !important;
            border-radius: 8px !important;
            transition: all 0.3s ease;
            margin: 0.25rem 0 !important;
        }}
        
        .stRadio > div > label:hover {{
            background-color: rgba(255, 255, 255, 0.2) !important;
        }}
        
        .stRadio > div > label[data-baseweb="radio"]:first-child {{
            background-color: rgba(255, 77, 77, 0.3) !important;
        }}
        
        /* تنسيق Slider - تأثير زجاجي */
           .stSlider {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .stSlider > div > div > div > div {{
            background: linear-gradient(90deg, rgba(255,77,77,0.8), rgba(255,0,0,0.7)) !important;
        }}
        
        .stSlider > div > div > div > div > div {{
            background: white !important;
            box-shadow: 0 0 0 2px rgba(255,77,77,0.8) !important;
        }}
        
        .stSlider > div > div > div > div > div:hover {{
            transform: scale(1.2);
        }}
        
        .stSlider > div > div > div > div > div:active {{
            transform: scale(1.3);
        }}
        
        .stSlider > div > label {{
            color: white !important;
            margin-bottom: 0.5rem !important;}}








            
            /* Divider line */
            hr {{
                border-color: rgba(255,255,255,0.2) !important;
                margin: 1.5rem 0 !important;
            }}
            
            /* Animation for title */
            @keyframes gradientPulse {{
                0% {{ background-position: 0% center; }}
                50% {{ background-position: 100% center; }}
                100% {{ background-position: 0% center; }}
            }}
            
            /* Info/Warning/Success boxes */
            .stAlert {{
                background-color: rgba(0,0,0,0.6) !important;
                border-left: 4px solid;
                border-radius: 12px !important;
                backdrop-filter: blur(5px);
            }}
        </style>
    """, unsafe_allow_html=True)

    with st.spinner("Initializing search engine..."):
        data = load_and_index()
    
    if data is None:
        st.error("""
            ❌ Failed to initialize search engine. Please check:
            1. Dataset folder exists and contains .txt files
            2. Files have readable content
            3. You have proper permissions
            """)
        st.stop()

    # UI Components
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header with animated gradient title
    st.markdown("""
        <div class="search-container">
            <h1 style='text-align:center; margin-bottom:0.5rem; font-size:2.5rem; background:linear-gradient(90deg, #ff0000, #ffffff, #ff0000); background-size:200% auto; color:transparent; -webkit-background-clip:text; background-clip:text; animation:gradientPulse 3s ease-in-out infinite;'>🔍 Doc Discover</h1>
            <p style='text-align:center; color:rgba(255,255,255,0.9); margin-top:0.5rem; font-size:1.15rem; font-weight:300; letter-spacing:0.3px; line-height:1.6;'>
                Search across your documents collection
            </p>
    """, unsafe_allow_html=True)
    
    # Search input with custom styling
    query = st.text_input(
        "", 
        placeholder="Enter your search query...", 
        key="search_input",
        label_visibility="collapsed"
    )
    
    # Search method selection with transparent radio buttons
    model = st.radio(
        "Search Method:",
        ["Document-Term Incidence", "Inverted Index", "TF-IDF with Cosine Similarity"],
        horizontal=True,
        index=2
    )
    
    # Results count slider (only for TF-IDF)
    if model == "TF-IDF with Cosine Similarity":
        k = st.slider("Number of results:", 1, 20, 5)
    
    # Search button
    if st.button("Search", key="search_btn", use_container_width=True):
        if not query.strip():
            st.warning("⚠️ Please enter a search query")
            st.stop()
        
        with st.spinner("🔍 Searching..."):
            try:
                query_terms = preprocess(query)
                st.info(f"**Processed terms:** {', '.join(query_terms)}")
                st.markdown("---")
                
                # Document-Term Incidence search
                if model == "Document-Term Incidence":
                    hits = search_term_doc_incidence(query_terms, data['term_doc_matrix'], data['vocabulary'])
                    if not hits:
                        st.warning("No documents matched all query terms")
                    else:
                        st.success(f"Found {len(hits)} matching documents")
                        for i in hits:
                            with st.container():
                                st.markdown(f"### 📄 {data['filenames'][i]}")
                                st.markdown(
                                    f'<div class="result-box">{get_snippet(data["raw_docs"][i], query_terms)}</div>',
                                    unsafe_allow_html=True
                                )
                
                # Inverted Index search
                elif model == "Inverted Index":
                    hits = search_inverted_index(query_terms, data['inverted_index'])
                    if not hits:
                        st.warning("No documents found")
                    else:
                        st.success(f"Found {len(hits)} matching documents")
                        for i in hits:
                            with st.container():
                                st.markdown(f"### 📂 {data['filenames'][i]}")
                                st.markdown(
                                    f'<div class="result-box">{get_snippet(data["raw_docs"][i], query_terms)}</div>',
                                    unsafe_allow_html=True
                                )
                
                # TF-IDF search
                else:
                    results = search_tfidf(data['tfidf_docs'], query, top_k=k)
                    if not results:
                        st.warning("No relevant documents found")
                    else:
                        st.success(f"Top {k} most relevant documents")
                        for i, score in results:
                            with st.container():
                                st.markdown(f"### 🏆 {data['filenames'][i]} (Score: {score:.3f})")
                                st.markdown(
                                    f'<div class="result-box">{get_snippet(data["raw_docs"][i], query_terms)}</div>',
                                    unsafe_allow_html=True
                                )
            
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")
                logging.exception("Search error")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()