import os, argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from load_data import load_data

class CustomTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_data_for_dual_transformer(X, y, tokenizer1, tokenizer2, max_len):
    if isinstance(X, pd.Series):
        X = X.tolist()

    encodings1 = tokenizer1(X, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    encodings2 = tokenizer2(X, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    
    labels = torch.tensor(y, dtype=torch.long)
    
    return CustomTextDataset({
            'input_ids1': encodings1['input_ids'],
            'attention_mask1': encodings1['attention_mask'],
            'input_ids2': encodings2['input_ids'],
            'attention_mask2': encodings2['attention_mask']
        }, labels)

class DualTransformerModel(nn.Module):
    def __init__(self, model_name1, model_name2, num_labels, hidden): 
        super(DualTransformerModel, self).__init__() 
        self.num_labels = num_labels 

        config1 = AutoConfig.from_pretrained(model_name1, output_attentions=True, output_hidden_states=True)
        config2 = AutoConfig.from_pretrained(model_name2, output_attentions=True, output_hidden_states=True)
        self.transformer1 = AutoModel.from_pretrained(model_name1, config=config1)
        self.transformer2 = AutoModel.from_pretrained(model_name2, config=config2)
        self.dropout = nn.Dropout(0.1) 
        
        combined_hidden_size = config1.hidden_size + config2.hidden_size

        self.custom_layers = nn.Sequential(
            nn.Linear(combined_hidden_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, self.num_labels)
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        outputs1 = self.transformer1(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.transformer2(input_ids=input_ids2, attention_mask=attention_mask2)

        sequence_output1 = self.dropout(outputs1.last_hidden_state[:, 0, :])
        sequence_output2 = self.dropout(outputs2.last_hidden_state[:, 0, :])

        combined_output = torch.cat((sequence_output1, sequence_output2), dim=1)
        logits = self.custom_layers(combined_output)
        return logits

def custom_collate_fn(batch):
    input_ids1 = torch.stack([item['input_ids1'] for item in batch])
    attention_mask1 = torch.stack([item['attention_mask1'] for item in batch])
    input_ids2 = torch.stack([item['input_ids2'] for item in batch])
    attention_mask2 = torch.stack([item['attention_mask2'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids1': input_ids1,
        'attention_mask1': attention_mask1,
        'input_ids2': input_ids2,
        'attention_mask2': attention_mask2,
        'labels': labels
    }

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
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
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    return total_loss / len(data_loader), accuracy

def main():
    parser = argparse.ArgumentParser(description="Choose hyperparameters for transformer training")
    parser.add_argument('--model1', type=str, required=True, help="Name of first pre-trained transformer model")
    parser.add_argument('--model2', type=str, required=True, help="Name of second pre-trained transformer model")
    parser.add_argument('--seq_length', type=int, required=True, help="Sequence length")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")
    parser.add_argument('--hidden', type=int, required=True, help="Number of neurons in hidden layer")
    parser.add_argument('--freeze', action='store_true', help="Enable initial freezing of transformer parameters")
    parser.add_argument('--folder', type=str, required=False, help="Folder name, where the model and tokenizer are going to be saved.")
    parser.add_argument('--save_name', type=str, required=False, help="Model will be saved with 'save_name'.pth in the folder")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Hyperparameters
    MODEL_NAME1 = args.model1
    MODEL_NAME2 = args.model2
    NUM_LABELS = 2
    MAX_SEQ_LENGTH = args.seq_length
    BATCH_SIZE = 16
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = 1e-5
    SAVE_FOLDER = './' + args.folder if args.folder else './results'
    MODEL_SAVE = os.path.join(SAVE_FOLDER, args.save_name + '.pth' if args.save_name else 'dual_transformer.pth')

    # TensorBoard SummaryWriter
    writer = SummaryWriter()

    train_path_neg = os.getcwd() + "/twitter-datasets/train_neg.txt"
    train_path_pos = os.getcwd() + "/twitter-datasets/train_pos.txt"
    test_path = os.getcwd() + "/twitter-datasets/test_data.txt"

    X_train, y_train, X_val, y_val, X_test = load_data(train_path_neg, train_path_pos, test_path, val_split=0.8, frac=1.0)

    tokenizer1 = AutoTokenizer.from_pretrained(MODEL_NAME1)
    tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAME2)

    train_dataset = preprocess_data_for_dual_transformer(X_train, y_train, tokenizer1, tokenizer2, MAX_SEQ_LENGTH)
    val_dataset = preprocess_data_for_dual_transformer(X_val, y_val, tokenizer1, tokenizer2, MAX_SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Training on {n_gpus} GPUs.")

    # Initialize model, optimizer, loss
    model = DualTransformerModel(model_name1=MODEL_NAME1, model_name2=MODEL_NAME2, num_labels=NUM_LABELS, hidden=args.hidden).to(device)
    
    if n_gpus > 1:
        model = nn.DataParallel(model)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCEWithLogitsLoss()

    if args.freeze:
        for param in model.module.transformer1.parameters():
            param.requires_grad = False
        for param in model.module.transformer2.parameters():
            param.requires_grad = False

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = eval_model(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # Unfreeze transformer weights after half of the epoch iterations
        if args.freeze:
            if epoch == NUM_EPOCHS // 2:
                for param in model.module.transformer1.parameters():
                    param.requires_grad = True
                for param in model.module.transformer2.parameters():
                    param.requires_grad = True

    os.makedirs(SAVE_FOLDER, exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE)
    tokenizer1.save_pretrained(SAVE_FOLDER)
    tokenizer2.save_pretrained(SAVE_FOLDER)

    writer.close()

    print("The dual transformer model was trained in the following configuration: ")
    print(args.model1, args.model2, args.seq_length, args.epochs, args.hidden, args.freeze)


if __name__ == "__main__":
    main()
