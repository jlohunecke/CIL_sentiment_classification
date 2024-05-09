from os import getcwd
import torch
import torch.nn as nn
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pdb
import pickle

from load_data import load_data
from preprocess import preprocess

# hyperparams
MAX_SEQUENCE_LENGTH = 40
MAX_VOCAB_SIZE = 20000
GRU_UNITS = 64
NUM_EPOCHS = 50
BATCH_SIZE = 5000
DROPOUT = 0.4
WEIGHT_DECAY = 1e-4
NUM_LAYERS = 1
# settings
RETRAIN = True
LOAD_PREPROCESSED = True
PREPROCESSED_PATH = getcwd() + "/twitter-datasets/preprocessed_data.pkl"
USE_JACOBS_PREPROCESSING = False
MODEL_SAVE_PATH = getcwd() + "/models/model_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load data
print("Loading and preprocessing data")
if LOAD_PREPROCESSED:
    with open(PREPROCESSED_PATH, 'rb') as f:
        preprocessed_data = pickle.load(f)
        X_train_, X_val_, X_test_ = preprocessed_data["X_train_"], preprocessed_data["X_val_"], preprocessed_data["X_test_"]
        X_train, X_val, X_test = preprocessed_data["X_train"], preprocessed_data["X_val"], preprocessed_data["X_test"]
        y_train, y_val = preprocessed_data["y_train"], preprocessed_data["y_val"]
else: 
    train_path_neg = getcwd() + "/twitter-datasets/train_neg.txt"
    train_path_pos = getcwd() + "/twitter-datasets/train_pos.txt"
    test_path = getcwd() + "/twitter-datasets/test_data.txt"
    X_train, y_train, X_val, y_val, X_test = load_data(train_path_neg, train_path_pos, test_path, val_split=0.8, frac=1.0)
    X_train_tokens, X_val_tokens, X_test_tokens = preprocess(X_train, X_val, X_test)
    X_train_ = [' '.join(tokens) for tokens in X_train_tokens]
    X_val_ = [' '.join(tokens) for tokens in X_val_tokens]
    X_test_ = [' '.join(tokens) for tokens in X_test_tokens]
    with open(PREPROCESSED_PATH, 'wb') as f:
        preprocessed_data = {
            "X_train_" : X_train_,
            "X_val_" : X_val_,
            "X_test_" : X_test_,
            "X_train" : X_train,
            "X_val" : X_val,
            "X_test" : X_test,
            "y_train" : y_train,
            "y_val" : y_val
        }
        pickle.dump(preprocessed_data, f)

# preprocess
def preprocess_for_rnn(X, y):
    if USE_JACOBS_PREPROCESSING:
        tokenized_sequences = [sequence.split(" ") for sequence in X]
    else:
        tokenizer = get_tokenizer("basic_english")
        tokenized_sequences = X.apply(tokenizer)
    all_tokens = [elt for sequence in tokenized_sequences for elt in sequence]
    counter = Counter(all_tokens)
    indexed_vocab = {elt[0] : idx for idx, elt in enumerate(counter.most_common(MAX_VOCAB_SIZE-1))}
    indexed_data = [torch.tensor([indexed_vocab.get(token, MAX_VOCAB_SIZE-1) for token in sequence]) for sequence in tokenized_sequences]
    padded_data = pad_sequence(indexed_data, batch_first=True).to(DEVICE)
    labels = torch.Tensor(y).to(DEVICE)
    return TensorDataset(padded_data[:, :MAX_SEQUENCE_LENGTH], labels)

if USE_JACOBS_PREPROCESSING:
    train_dataset = preprocess_for_rnn(X_train_, y_train)
    val_dataset = preprocess_for_rnn(X_val_, y_val)
else:
    train_dataset = preprocess_for_rnn(X_train, y_train)
    val_dataset = preprocess_for_rnn(X_val, y_val)

# define RNN
class SentimentAnalysisRNN(nn.Module):
    def __init__(self, max_vocab_size, embedding_dim, gru_units):
        super(SentimentAnalysisRNN, self).__init__()        
        self.embedding = nn.Embedding(max_vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, gru_units, dropout=DROPOUT, batch_first=True, num_layers=NUM_LAYERS)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(gru_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        output = self.dropout(output[:, -1, :])  # Taking the last hidden state
        output = self.fc(output)
        output = self.sigmoid(output)
        return output.squeeze(1)  # Squeeze to remove the extra dimension
    
# train RNN
print("Training started")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = SentimentAnalysisRNN(MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH, GRU_UNITS).to(DEVICE)
if RETRAIN:
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
    for epoch in range(NUM_EPOCHS):
        # perform 
        model.train()  # Set the model to training mode
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels.float()) # compute loss
            loss.backward() # Backward pass   
            optimizer.step() # Update the parameters        
            train_loss += loss.item()
            predicted = torch.round(outputs)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        train_loss /= len(train_dataloader)
        train_accuracy = train_correct / train_total
        # perform validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                predicted = torch.round(outputs)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / val_total
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, '
            f'Train Loss: {train_loss:.4f}, '
            f'Train Accuracy: {train_accuracy:.2%}, '
            f'Val Loss: {val_loss:.4f}, '
            f'Val Accuracy: {val_accuracy:.2%}')
    print("Training finished")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model weights saved to:", MODEL_SAVE_PATH)
else:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print("Loaded pretrained weights from:", MODEL_SAVE_PATH)