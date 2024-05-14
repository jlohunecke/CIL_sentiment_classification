import numpy as np
import gensim.downloader as api
from scipy.sparse import csr_matrix
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


class GloveEmbedder():

    def __init__(self, embed_size=25):
        self.embed_size = embed_size
        self.glove_path = f'../glove_files/glove.twitter.27B.{self.embed_size}d.txt'
        self.embedder = self.load_glove()

    def load_glove(self):

        glove_model = api.load(f"glove-twitter-{self.embed_size}")
        #glove2word2vec(glove_input_file=self.glove_path, word2vec_output_file="gensim_glove_vectors.txt")
        #glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
        return glove_model

    def _embed_token(self, token):
        try:
            return self.embedder[token]
        except KeyError:
            return np.zeros(self.embed_size)

    def _embed_sentence(self, sentence):
        embeddings = np.array([self._embed_token(token) for token in sentence.split()])

        try:
            out = np.mean(embeddings, axis=0)
            return out
        except:
            return np.zeros(self.embed_size)

    def transform(self, sentences):
        embeddings = []
        for sentence in sentences:

            if sentence == "":
                emb = np.zeros(self.embed_size)
            else:
                emb = self._embed_sentence(sentence)

            #TODO: check why we get nans - probably because of mean
            if np.isnan(emb).any():
                emb = np.zeros(self.embed_size)

            embeddings.append(emb)

        return csr_matrix(np.vstack(embeddings))
