import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
import gensim.downloader as api
from scipy.sparse import csr_matrix
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(train_path_neg, train_path_pos, test_path, val_split=0.9, frac=1):
    """
    Loads and splits the data into training, validation, and test sets.

    Parameters:
        train_path_neg (str): The file path to the negative training data.
        train_path_pos (str): The file path to the positive training data.
        test_path (str): The file path to the test data.
        val_split (float): The proportion of the data to use for training. The rest is used for validation. Default is 0.9.
        frac (float): Fraction of the data to sample. Default is 1 (use all data).

    Returns:
        tuple: A tuple containing the following:
            X_train (pd.Series): The training tweets.
            y_train (pd.Series): The training labels.
            X_val (pd.Series): The validation tweets.
            y_val (pd.Series): The validation labels.
            X_test (pd.Series): The test tweets.
            y_test_dummy (pd.Series): Dummy labels for the test data.
    """
    data_neg = pd.read_fwf(train_path_neg, header=None, names=["tweet"])
    data_neg["label"] = 0
    train_neg = data_neg[:int(val_split * len(data_neg))]
    val_neg = data_neg[int(val_split * len(data_neg)):]

    data_pos = pd.read_fwf(train_path_pos, header=None, names=["tweet"])
    data_pos["label"] = 1
    train_pos = data_pos[:int(val_split * len(data_pos))]
    val_pos = data_pos[int(val_split * len(data_pos)):]

    train = pd.concat([train_neg, train_pos], ignore_index=True).sample(frac=frac).reset_index(drop=True)
    X_train = train["tweet"].squeeze()
    y_train = train["label"].squeeze()

    val = pd.concat([val_neg, val_pos], ignore_index=True).sample(frac=frac).reset_index(drop=True)
    X_val = val["tweet"].squeeze()
    y_val = val["label"].squeeze()

    X_test = pd.read_fwf(test_path, header=None, dtype=str)[0]
    X_test = X_test.apply(lambda row: "".join(row.split(",")[1:])) # remove row indices from tweets
    y_test_dummy = pd.Series([-1] * len(X_test))
    
    return X_train, y_train, X_val, y_val, X_test, y_test_dummy


class GloveEmbedder():
    """
    A class to load GloVe embeddings and transform sentences into their corresponding vector representations.
    
    Attributes:
        embed_size (int): The dimensionality of the GloVe embeddings.
        glove_path (str): The file path to the GloVe embeddings.
        embedder (KeyedVectors): The GloVe model.
    """

    def __init__(self, embed_size=25):
        """
        Initializes the GloveEmbedder with the specified embedding size.
        
        Args:
            embed_size (int): The dimensionality of the GloVe embeddings. Default is 25.
        """
        self.embed_size = embed_size
        self.glove_path = f'../glove_files/glove.twitter.27B.{self.embed_size}d.txt'
        self.embedder = self.load_glove()

    def load_glove(self):
        """
        Loads the GloVe model from gensim's downloader.

        Returns:
            The loaded GloVe model.
        """
        glove_model = api.load(f"glove-twitter-{self.embed_size}")
        return glove_model

    def _embed_token(self, token):
        """
        Gets the embedding for a single token.

        Args:
            token (str): The token to embed.

        Returns:
            numpy.ndarray: The embedding vector for the token. Returns a zero vector if the token is not in the model.
        """
        try:
            return self.embedder[token]
        except KeyError:
            return np.zeros(self.embed_size)

    def _embed_sentence(self, sentence):
        """
        Gets the embedding for a sentence by averaging the embeddings of its tokens.

        Args:
            sentence (str): The sentence to embed.

        Returns:
            numpy.ndarray: The embedding vector for the sentence. Returns a zero vector if the sentence is empty or cannot be embedded.
        """
        embeddings = np.array([self._embed_token(token) for token in sentence.split()])

        try:
            out = np.mean(embeddings, axis=0)
            return out
        except:
            return np.zeros(self.embed_size)

    def transform(self, sentences):
        """
        Transforms a list of sentences into a sparse matrix of their embeddings.

        Args:
            sentences (list of str): The sentences to transform.

        Returns:
            csr_matrix: A sparse matrix where each row corresponds to the embedding of a sentence.
        """
        embeddings = []
        for sentence in sentences:

            if sentence == "":
                emb = np.zeros(self.embed_size)
            else:
                emb = self._embed_sentence(sentence)

            if np.isnan(emb).any():
                emb = np.zeros(self.embed_size)

            embeddings.append(emb)

        return csr_matrix(np.vstack(embeddings))


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

    # remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = tokens.apply(lambda x: [word for word in x if word not in stop_words])

    return tokens
