import numpy as np
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class Embedder:

    def __init__(self):

        self.pipeline = self.pipeline = Pipeline([
            ('bow', CountVectorizer()),  # strings to token integer counts
            ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ])
        pass

    def fit(self, X_train_tokens):
        X_train_tokens = X_train_tokens.astype(str)
        self.pipeline.fit(X_train_tokens)
        return self.pipeline

    def get_tweet_vector(self, tweet):
        vector_size = self.embedder.vector_size
        token_embeddings = []

        for token in tweet:
            try:
                token_embeddings.append(self.embedder.wv[token])
            except KeyError:
                pass

        if token_embeddings:
            return np.mean(token_embeddings, axis=0)
        else:
            return np.zeros(vector_size)
