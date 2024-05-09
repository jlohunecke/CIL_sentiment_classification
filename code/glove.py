from numpy import genfromtxt
import pdb

def get_glove_embeddings(stanford_path):
    stanford_labels = genfromtxt(stanford_path, delimiter=' ', encoding="utf8", usecols=(0), dtype=None)
    stanford_data = genfromtxt(stanford_path, delimiter=' ', encoding="utf8")
    stanford_embeddings = {stanford_labels[i] : stanford_data[i, 1:] for i in range(stanford_labels.shape[0])}
    return stanford_embeddings

glove_embeddings = get_glove_embeddings('glove_data/glove.twitter.27B.25d.txt')