import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer



class Embedder:

    def __init__(self):

        self.model = None

        pass

    def fit(self, X_tokens):

        # Train word2vec model
        self.model = Word2Vec(X_tokens, vector_size=100, window=5, min_count=1, workers=4)

        pass

    def embed(self, X_tokens):

        preprocessed_sentences = [' '.join(tokens) for tokens in X_tokens]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
        tfidf_array = tfidf_matrix.toarray()

        # Get word embeddings
        word_embeddings = []
        for word in vectorizer.get_feature_names_out():
            try:
                word_embeddings.append(self.model.wv[word])
            except:
                pass

        sentence_embeddings = []
        for row in tfidf_array:
            sentence_embedding = np.dot(row, word_embeddings)
            sentence_embeddings.append(sentence_embedding)

        sentence_embeddings = np.array(sentence_embeddings)
        #norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        #normalized_embeddings = sentence_embeddings / norms

        return sentence_embeddings
