from os import getcwd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model, load_model
from sklearn.metrics import accuracy_score, classification_report

from load_data import load_data
from preprocess import preprocess

RETRAIN = False
model_path = getcwd() + "/../models/"

# load data
train_path_neg = getcwd() + "/twitter-datasets/train_neg.txt"
train_path_pos = getcwd() + "/twitter-datasets/train_pos.txt"
test_path = getcwd() + "/twitter-datasets/test_data.txt"
X_train, y_train, X_val, y_val, X_test = load_data(train_path_neg, train_path_pos, test_path, frac=0.2)

# preprocess 
max_sequence_length = 40
max_vocab_size = 20000
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<UNK>')
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
train_inputs = pad_sequences(train_sequences, maxlen=max_sequence_length)
tokenizer.fit_on_texts(X_val)
val_sequences = tokenizer.texts_to_sequences(X_val)
val_inputs = pad_sequences(val_sequences, maxlen=max_sequence_length)

# train
if RETRAIN:
    embedding_dim = 50
    gru_units = 64
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
        tf.keras.layers.GRU(units=gru_units, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    batch_size = 32
    epochs = 1
    model.fit(train_inputs, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    save_model(model, model_path + "gru_model.h5")
else:
    model = load_model(model_path + "gru_model.h5")

# validate
predictions = model.predict(val_inputs)
binary_predictions = (predictions > 0.5).astype(int)
accuracy = accuracy_score(y_val, binary_predictions)
report = classification_report(y_val, binary_predictions)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

print()