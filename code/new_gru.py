import pandas as pd
import pdb
from os import getcwd
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split

from load_data import load_data

# Parameters
vocab_size = 20000
embedding_dim = 100
max_length = 40
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
dropout = 0.5

# read and clean data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

train_path_neg = getcwd() + "/twitter-datasets/train_neg.txt"
train_path_pos = getcwd() + "/twitter-datasets/train_pos.txt"
test_path = getcwd() + "/twitter-datasets/test_data.txt"
X_train, y_train, X_val, y_val, X_test = load_data(train_path_neg, train_path_pos, test_path, val_split=0.8, frac=1.0)

data = {"text" : X_train.values.tolist(), "label" : y_train.values.tolist()}
data["text"] += X_val.values.tolist()
data["label"] += y_val.values.tolist()
df = pd.DataFrame(data)
df['text'] = df['text'].apply(clean_text)

# Tokenize the text data and convert to sqeuences of uniform length
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(df['text'])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
labels = df['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# use glove embeddings
embedding_index = {}
with open(getcwd() + '/glove_data/glove.twitter.27B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# build model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
    Bidirectional(GRU(64, return_sequences=True)),
    Dropout(dropout),
    Bidirectional(GRU(32)),
    Dropout(dropout),
    Dense(32, activation='relu'),
    Dropout(dropout),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# train and evaluate
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=2)
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Accuracy: {accuracy:.4f}')