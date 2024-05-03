from scipy.sparse import load_npz
from os import getcwd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from load_data import load_data
from preprocess import preprocess

train_path_neg = getcwd() + "/twitter-datasets/train_neg.txt"
train_path_pos = getcwd() + "/twitter-datasets/train_pos.txt"
test_path = getcwd() + "/twitter-datasets/test_data.txt"
X_train, y_train, X_val, y_val, X_test = load_data(train_path_neg, train_path_pos, test_path, frac=0.2)

max_vocab_size = 10000
max_sequence_length = 20

tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))




print()

# preprocessed_path = getcwd() + "/preprocessed_data/"
# X_train_c_vec = load_npz(preprocessed_path + "X_train_c_vec.npz")
# X_train_t_vec = load_npz(preprocessed_path + "X_train_t_vec.npz")
# X_val_c_vec = load_npz(preprocessed_path + "X_val_c_vec.npz")
# X_test_c_vec = load_npz(preprocessed_path + "X_test_c_vec.npz")
# X_val_t_vec = load_npz(preprocessed_path + "X_val_t_vec.npz")
# X_test_t_vec = load_npz(preprocessed_path + "X_test_t_vec.npz")
# y_train = np.load(preprocessed_path + "y_train.npz")["data"]
# y_val = np.load(preprocessed_path + "y_val.npz")["data"]

# embedding_dim = 50
# gru_units = 64
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
#     tf.keras.layers.GRU(units=gru_units, dropout=0.2, recurrent_dropout=0.2),
#     tf.keras.layers.Dense(units=1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()

# print()