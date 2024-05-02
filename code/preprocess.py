import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess(tweets_train, tweets_val, tweets_test):

    # Preprocess tweets
    prep_tweets_train = preprocess_tweets(tweets_train)
    prep_tweets_val = preprocess_tweets(tweets_val)
    prep_tweets_test = preprocess_tweets(tweets_test)

    # Tokenize tweets
    tokens_train = prep_tweets_train.apply(word_tokenize)
    tokens_val = prep_tweets_val.apply(word_tokenize)
    tokens_test = prep_tweets_test.apply(word_tokenize)

    # Preprocess tokens
    prep_tokens_train = preprocess_tokens(tokens_train)
    prep_tokens_val = preprocess_tokens(tokens_val)
    prep_tokens_test = preprocess_tokens(tokens_test)

    return prep_tokens_train, prep_tokens_val, prep_tokens_test


def preprocess_tweets(tweets):

    # Convert to string and lowercase
    tweets = tweets.apply(str).apply(str.lower)

    # Define patterns to remove
    patterns = [
        (re.compile(r'<[^>]*>'), ' '), # HTML tags
        (re.compile(r'^RT[\s]+'), ' '), # RT
        (re.compile(r'https?:\/\/.*[\r\n]*'), ' '), # URLs
        (re.compile(r'#'), ''), # Hashtags
        (re.compile(r'@[A-Za-z0-9]+'), ' '), # Mentions
        (re.compile(r'[^\w\s]'), ' '), # Punctuation
        (re.compile(r'\d+'), ' '), # Numbers
        (re.compile(r'\s+'), ' '), # Whitespace
    ]

    # Apply patterns
    for pattern, replacement in patterns:
        tweets = tweets.apply(lambda x: pattern.sub(replacement, x))

    return tweets


def preprocess_tokens(tokens):

    # remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = tokens.apply(lambda x: [word for word in x if word not in stop_words])

    # remove punctuation
    tokens = tokens.apply(lambda x: [word for word in x if word not in string.punctuation])

    # remove non-alphanumeric characters
    tokens = tokens.apply(lambda x: [word for word in x if word.isalnum()])

    # stem words
    stemmer = PorterStemmer()
    tokens = tokens.apply(lambda x: [stemmer.stem(word) for word in x])

    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = tokens.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    return tokens



