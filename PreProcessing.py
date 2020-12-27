# nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

PONCTUATION = set(string.punctuation)
STOP_WORDS = stopwords.words("english")

STOP_WORDS.extend(PONCTUATION)

def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens_without_stopwords = [word for word in tokens if word not in STOP_WORDS]
    return tokens_without_stopwords

def lemmatize(tokens):
    lemma = WordNetLemmatizer()
    tokens_lemmatize = [lemma.lemmatize(word,'a') for word in tokens]
    tokens_lemmatize = [lemma.lemmatize(word,'v') for word in tokens]
    tokens_lemmatize = [lemma.lemmatize(word,'n') for word in tokens]
    return tokens_lemmatize

def preProcessing(text):
    tokens = tokenize(text)
    tokens_lemmatize = lemmatize(tokens)
    return " ".join(tokens_lemmatize)