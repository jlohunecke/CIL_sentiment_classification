import numpy as np
import gensim.downloader as api
from scipy.sparse import csr_matrix
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


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
