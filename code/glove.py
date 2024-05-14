import numpy as np
import torch
import pdb

def get_glove_embeddings(stanford_path):
    print("loading glove embeddings..", end=" ")
    stanford_labels = np.genfromtxt(stanford_path, delimiter=' ', encoding="utf8", usecols=(0), dtype=None)
    stanford_data = np.genfromtxt(stanford_path, delimiter=' ', encoding="utf8")
    stanford_embeddings = {stanford_labels[i] : stanford_data[i, 1:] for i in range(stanford_labels.shape[0])}
    print("done")
    return stanford_embeddings

def create_embedding_matrix(vocab, glove_path, default, print_missing_words=False):
    glove_embeddings = get_glove_embeddings(glove_path)
    embedding_matrix = np.zeros((len(vocab)+1, len(glove_embeddings[next(iter(glove_embeddings.keys()))])))
    words_not_found = []
    for word, index in vocab.items():
        embedding_matrix[index] = glove_embeddings.get(word, default)
        if glove_embeddings.get(word) is None:
            words_not_found.append(word)
    if print_missing_words:
        print("Words not found in GloVe embedding:", words_not_found)
    return torch.from_numpy(embedding_matrix)