import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


# Download required NLTK resources (run once)
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Initialize tools
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
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
    """
    # 1. Case folding
    text = text.lower()

    # 2. Tokenization (splits into words)
    tokens = word_tokenize(text)

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
