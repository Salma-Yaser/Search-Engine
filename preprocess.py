import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


# Initialize tools
try:
    # Initialize stopwords - with fallback for when deployment doesn't have NLTK data
    try:
        STOPWORDS = set(stopwords.words('english'))
    except LookupError:
        # Fallback to a basic set of English stopwords if NLTK data is not available
        STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                    'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
                    'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
                    'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about', 'against', 'between',
                    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
                    'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'my', 'mine', 'your', 'yours',
                    'his', 'her', 'hers', 'its', 'their', 'theirs', 'our', 'ours'}
        
    # Initialize lemmatizer and stemmer with fallbacks
    try:
        lemmatizer = WordNetLemmatizer()
    except LookupError:
        # Simple lemmatizer function as fallback
        lemmatizer = type('', (), {'lemmatize': lambda self, word, pos='n': word})()
    
    stemmer = PorterStemmer()  # PorterStemmer doesn't require additional NLTK data
except Exception as e:
    # Final fallback if anything goes wrong
    print(f"Error initializing NLP tools: {e}")
    STOPWORDS = set()
    lemmatizer = type('', (), {'lemmatize': lambda self, word, pos='n': word})()
    stemmer = PorterStemmer()


def preprocess(text: str) -> list[str]:
    """
    Perform full preprocessing on input text:
      1. Case folding (lowercasing)
      2. Tokenization
      3. Remove punctuation
      4. Stop-word removal
      5. Lemmatization
      6. Stemming

    Returns a list of processed tokens.
    """    # 1. Case folding
    text = text.lower()

    # 2. Tokenization (splits into words)
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Simple fallback tokenization if NLTK data isn't available
        tokens = text.split()

    processed_tokens = []
    for token in tokens:
        # 3. Remove non-alphanumeric tokens/punctuation
        token = re.sub(r"[^\w\s]", "", token)
        if not token:
            continue

        # 4. Stop-word removal
        if token in STOPWORDS:
            continue

        # 5. Lemmatization
        lemma = lemmatizer.lemmatize(token)

        # 6. Stemming (after lemmatization)
        #stem = stemmer.stem(lemma)

        processed_tokens.append(lemma)

    return processed_tokens
