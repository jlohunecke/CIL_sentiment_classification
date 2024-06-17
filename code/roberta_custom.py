import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW
from torch.utils.data import DataLoader

from load_data import load_data

class CustomTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_data_for_roberta(X, y, tokenizer):
    if isinstance(X, pd.Series):
        X = X.tolist()

    encodings = tokenizer(X, truncation=True, padding=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
    
    labels = torch.tensor(y, dtype=torch.long)
    
    return CustomTextDataset(encodings, labels)

class CustomRoberta(nn.Module):
    def __init__(self, model_name, num_labels): 
        super(CustomRoberta, self).__init__() 
        self.num_labels = num_labels 

        self.roberta = AutoModel.from_pretrained(model_name, config=AutoConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True))
        self.dropout = nn.Dropout(0.1) 
        
        ## this added architecture is variable and can be modified to best fit the data
        self.custom_layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.custom_layers(sequence_output)
        return logits

def custom_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    return total_loss / len(data_loader), accuracy

# TO surpress warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hyperparameters
MODEL_NAME = 'roberta-base'
NUM_LABELS = 2
MAX_SEQ_LENGTH = 80
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
MODEL_SAVE_PATH = './results/initial_freeze.pth'
TOKENIZER_SAVE_PATH = './results'

# Adapted from Kai's GRU implementation
train_path_neg = os.getcwd() + "/twitter-datasets/train_neg.txt"
train_path_pos = os.getcwd() + "/twitter-datasets/train_pos.txt"
test_path = os.getcwd() + "/twitter-datasets/test_data.txt"

X_train, y_train, X_val, y_val, X_test = load_data(train_path_neg, train_path_pos, test_path, val_split=0.8, frac=1.0)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = preprocess_data_for_roberta(X_train, y_train, tokenizer)
val_dataset = preprocess_data_for_roberta(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Training on {n_gpus} GPUs.")

# Initialize model, optimizer, loss
model = CustomRoberta(model_name=MODEL_NAME, num_labels=NUM_LABELS).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# Freeze all the parameters of the roberta model initially
for param in model.roberta.parameters():
    param.requires_grad = False


# Simple training loop
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_accuracy = eval_model(model, val_loader, loss_fn, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


    # Unfreeze RoBERTa weights after half of the epoch iterations
    if epoch == NUM_EPOCHS // 2:
        for param in model.roberta.parameters():
            param.requires_grad = True

# Save the model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)


