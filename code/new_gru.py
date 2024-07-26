import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
from os import getcwd

from load_data import load_data
from transformers import AutoTokenizer, PreTrainedTokenizerFast

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Initializes the dataset with texts, labels, tokenizer, and max length.

        Args:
            texts (list): List of texts.
            labels (list): List of labels.
            tokenizer (PreTrainedTokenizerFast): Tokenizer for encoding texts.
            max_length (int): Maximum length of tokenized sequences.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, embedding_matrix):
        """
        Initializes the GRU model with specified parameters.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of embeddings.
            hidden_dim (int): Dimension of GRU hidden layers.
            output_dim (int): Dimension of the output layer.
            n_layers (int): Number of GRU layers.
            bidirectional (bool): If the GRU is bidirectional.
            dropout (float): Dropout rate.
            embedding_matrix (torch.Tensor): Pre-trained embedding matrix.
        """
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the GRU model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Model output logits.
        """
        embedded = self.dropout(self.embedding(input_ids))
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        output = self.fc(hidden)
        return output

def clean_text(text):
    """
    Cleans the input text by removing HTML tags, non-alphabetic characters, and extra whitespace.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

def load_glove_embeddings(glove_path, word_index, embedding_dim, vocab_size):
    """
    Loads pre-trained GloVe embeddings.

    Args:
        glove_path (str): Path to the GloVe file.
        word_index (dict): Word index from the tokenizer.
        embedding_dim (int): Dimension of embeddings.
        vocab_size (int): Size of the vocabulary.

    Returns:
        torch.Tensor: Embedding matrix.
    """
    embedding_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i < vocab_size:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return torch.tensor(embedding_matrix, dtype=torch.float)

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for the training data.
        loss_fn (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to run the training on.

    Returns:
        tuple: Training accuracy and average loss.
    """
    model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels.unsqueeze(1))

        correct_predictions += torch.sum((outputs > 0.5) == labels.unsqueeze(1)).item()
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
    """
    Evaluates the model.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation data.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple: Evaluation accuracy and average loss.
    """
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels.unsqueeze(1))

            correct_predictions += torch.sum((outputs > 0.5) == labels.unsqueeze(1)).item()
            losses.append(loss.item())

    return correct_predictions / len(data_loader.dataset), np.mean(losses)

def main():
    # Parameters
    vocab_size = 20000
    embedding_dim = 100
    max_length = 40
    dropout = 0.5
    hidden_dim = 64
    n_layers = 2
    bidirectional = True
    output_dim = 1
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-3

    # Read and clean data
    train_path_neg = getcwd() + "/twitter-datasets/train_neg.txt"
    train_path_pos = getcwd() + "/twitter-datasets/train_pos.txt"
    test_path = getcwd() + "/twitter-datasets/test_data.txt"
    X_train, y_train, X_val, y_val, X_test = load_data(train_path_neg, train_path_pos, test_path, val_split=0.8, frac=1.0)

    data = {"text": X_train.values.tolist(), "label": y_train.values.tolist()}
    data["text"] += X_val.values.tolist()
    data["label"] += y_val.values.tolist()
    df = pd.DataFrame(data)
    df['text'] = df['text'].apply(clean_text)

    # Tokenize the text data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.model_max_length = max_length
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    sequences = tokenizer(df['text'].tolist(), padding='max_length', truncation=True, max_length=max_length)
    padded_sequences = np.array(sequences['input_ids'])
    labels = df['label'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    # Load GloVe embeddings
    glove_path = getcwd() + '/glove_data/glove.twitter.27B.100d.txt'
    embedding_matrix = load_glove_embeddings(glove_path, tokenizer.vocab, embedding_dim, vocab_size)

    # Create datasets and dataloaders
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model, loss function, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, embedding_matrix).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and evaluate the model
    for epoch in range(num_epochs):
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')

    test_acc, test_loss = eval_model(model, test_loader, loss_fn, device)
    print(f'Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    main()
