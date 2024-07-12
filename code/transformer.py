import os
import argparse
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
import pdb
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

def preprocess_data_for_transformer(X, y, tokenizer, max_len):
    if isinstance(X, pd.Series):
        X = X.tolist()

    encodings = tokenizer(X, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    
    labels = torch.tensor(y, dtype=torch.long)
    
    return CustomTextDataset(encodings, labels)

class CustomTransformer(nn.Module):
    def __init__(self, model_name, num_labels, hidden): 
        super(CustomTransformer, self).__init__() 
        self.num_labels = num_labels 

        config = AutoConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(0.1) 
        
        ## this added architecture is variable and can be modified to best fit the data
        self.custom_layers = nn.Sequential(
            nn.Linear(config.hidden_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, self.num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
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

def predict_test(test_loader, model, save_path, device):
    model.eval()
    output = torch.zeros((len(test_loader.dataset)), 1)
    print("predicting on test set...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=int(len(test_loader.dataset) // test_loader.batch_size)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            probabilities = model(input_ids, attention_mask)
            predictions = torch.argmax(probabilities, dim=1)
            output[i*test_loader.batch_size:(i+1)*test_loader.batch_size, 0] = predictions
    df = pd.DataFrame(output.int().numpy(), columns=['Prediction'])
    df['Prediction'] = df['Prediction'].replace(0, -1)
    df['Id'] = df.reset_index(drop=True).index + 1
    df.to_csv(save_path, index=False)
    print("saved test set predictions to", save_path)

def load_model(model_path, tokenizer_path, model_name, num_labels, hidden, device):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load the model
    model = CustomTransformer(model_name=model_name, num_labels=num_labels, hidden=hidden)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Choose hyperparameters for transformer training")
    parser.add_argument('--model', type=str, required=True, help="Name of pre-trained transformer model")
    parser.add_argument('--seq_length', type=int, required=True, help="Sequence length")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")
    parser.add_argument('--hidden', type=int, required=True, help="Number of neurons in hidden layer")
    parser.add_argument('--freeze', action='store_true', help="Enable initial freezing of transformer parameters")
    parser.add_argument('--folder', type=str, required=False, help="Folder name, where the model and tokenizer are going to be saved.")
    parser.add_argument('--save_name', type=str, required=False, help="Model will be saved with 'save_name'.pth in the folder")
    parser.add_argument('--inference_name', type=str, required=False, help="After training, model will predict results for the test dataset and save them to 'inference_name.csv' in the save folder")
    parser.add_argument('--batch', type=int, help="batch size, default is 16")
    parser.add_argument('--load_model', type=str, help="Path to load the model for inference")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Hyperparameters
    MODEL_NAME = args.model
    NUM_LABELS = 2
    MAX_SEQ_LENGTH = args.seq_length
    BATCH_SIZE = args.batch if args.batch else 16
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = 1e-5
    SAVE_FOLDER = './' + args.folder if args.folder else './results'
    MODEL_SAVE = os.path.join(SAVE_FOLDER, args.save_name + '.pth' if args.save_name else 'single_transformer.pth')
    INFERENCE_SAVE = os.path.join(SAVE_FOLDER, args.inference_name + '.csv') if args.inference_name else None

    # TensorBoard SummaryWriter
    writer = SummaryWriter()

    # Adapted from Kai's GRU implementation
    train_path_neg = os.getcwd() + "/twitter-datasets/train_neg.txt"
    train_path_pos = os.getcwd() + "/twitter-datasets/train_pos.txt"
    test_path = os.getcwd() + "/twitter-datasets/test_data.txt"

    X_train, y_train, X_val, y_val, X_test, y_test_dummy = load_data(train_path_neg, train_path_pos, test_path, val_split=0.8, frac=1.0)
    # pdb.set_trace()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("mps")
    n_gpus = torch.cuda.device_count()
    print(f"Training on {n_gpus} GPUs.")

    # Initialize model, optimizer, loss
    model = CustomTransformer(model_name=MODEL_NAME, num_labels=NUM_LABELS, hidden=args.hidden).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    train_dataset = preprocess_data_for_transformer(X_train, y_train, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = preprocess_data_for_transformer(X_val, y_val, tokenizer, MAX_SEQ_LENGTH)
    test_dataset = preprocess_data_for_transformer(X_test, y_test_dummy, tokenizer, MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    os.makedirs(SAVE_FOLDER, exist_ok=True)
        
    if args.freeze:
        for param in model.transformer.parameters():
            param.requires_grad = False

    if args.load_model:
        model, tokenizer = load_model(args.load_model, SAVE_FOLDER, MODEL_NAME, NUM_LABELS, args.hidden, device)
        print("Model loaded successfully.")
    else:
        best_val_loss = float('inf')  # Initialize the best validation loss to infinity

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, val_accuracy = eval_model(model, val_loader, loss_fn, device)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)

            # Save the model if validation loss has decreased
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_SAVE)
                print(f"Model saved with validation loss: {val_loss:.4f}")

            # Unfreeze transformer weights after half of the epoch iterations
            if args.freeze:
                if epoch == NUM_EPOCHS // 2:
                    for param in model.transformer.parameters():
                        param.requires_grad = True

        tokenizer.save_pretrained(SAVE_FOLDER)

        # Make predictions for test data
        # pdb.set_trace()
        predict_test(test_loader, model, INFERENCE_SAVE, device)

    writer.close()

    print("The custom transformer was trained in the following configuration: ")
    print(args.model, args.seq_length, args.epochs, args.hidden, args.freeze)

if __name__ == "__main__":
    main()

    # python code/transformer.py --model "bert-large-uncased" --seq_length 50 --epochs 40 --hidden 512 --freeze
    ### for tensorboard output, connect via:
    # ssh -L 6006:localhost:6006 username@remote_server_address
    ### after starting the training script:
    # tensorboard --logdir=runs --port=6006
    ### on local machine, open:
    # http://localhost:6006
