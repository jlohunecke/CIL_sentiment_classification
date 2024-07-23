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
    def __init__(self, encodings_list, labels):
        self.encodings_list = encodings_list
        self.labels = labels

    def __getitem__(self, idx):
        item = {f'input_ids_{i}': encodings['input_ids'][idx] for i, encodings in enumerate(self.encodings_list)}
        item.update({f'attention_mask_{i}': encodings['attention_mask'][idx] for i, encodings in enumerate(self.encodings_list)})
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_data_for_transformer(X, y, tokenizers, max_len):
    if isinstance(X, pd.Series):
        X = X.tolist()

    encodings_list = []
    for tokenizer in tokenizers:
        encodings = tokenizer(X, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        encodings_list.append(encodings)

    labels = torch.tensor(y, dtype=torch.long)
    return CustomTextDataset(encodings_list, labels)


class MultiModelTransformer(nn.Module):
    def __init__(self, model_names, num_labels, hidden_depth, hidden_width):
        super(MultiModelTransformer, self).__init__()
        self.num_labels = num_labels

        self.transformers = nn.ModuleList([AutoModel.from_pretrained(model_name) for model_name in model_names])
        hidden_size = sum([AutoConfig.from_pretrained(model_name).hidden_size for model_name in model_names])
        
        self.dropout = nn.Dropout(0.1)
        
        hidden_layers = []
        for i in range(hidden_depth):
            if i == 0:
                hidden_layers.append(nn.Linear(hidden_size, hidden_width))
            else:
                hidden_layers.append(nn.Linear(hidden_width, hidden_width))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(0.1))
        # Add the final layer
        hidden_layers.append(nn.Linear(hidden_width, self.num_labels))

        self.custom_layers = nn.Sequential(*hidden_layers)

    def forward(self, input_ids_list, attention_mask_list):
        transformer_outputs = [transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :] 
                               for transformer, input_ids, attention_mask in zip(self.transformers, input_ids_list, attention_mask_list)]
        concatenated_output = torch.cat(transformer_outputs, dim=-1)
        sequence_output = self.dropout(concatenated_output)
        logits = self.custom_layers(sequence_output)
        return logits

def custom_collate_fn(num_models):
    def collate_fn(batch):
        input_ids_list = [torch.stack([item[f'input_ids_{i}'] for item in batch]) for i in range(num_models)]
        attention_mask_list = [torch.stack([item[f'attention_mask_{i}'] for item in batch]) for i in range(num_models)]
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids_list': input_ids_list,
            'attention_mask_list': attention_mask_list,
            'labels': labels
        }
    return collate_fn


def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        input_ids_list = [input_ids.to(device) for input_ids in batch['input_ids_list']]
        attention_mask_list = [attention_mask.to(device) for attention_mask in batch['attention_mask_list']]
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids_list, attention_mask_list)
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
            input_ids_list = [input_ids.to(device) for input_ids in batch['input_ids_list']]
            attention_mask_list = [attention_mask.to(device) for attention_mask in batch['attention_mask_list']]
            labels = batch['labels'].to(device)

            outputs = model(input_ids_list, attention_mask_list)
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
            input_ids_list = [input_ids.to(device) for input_ids in batch['input_ids_list']]
            attention_mask_list = [attention_mask.to(device) for attention_mask in batch['attention_mask_list']]
            probabilities = model(input_ids_list, attention_mask_list)
            predictions = torch.argmax(probabilities, dim=1)
            output[i*test_loader.batch_size:(i+1)*test_loader.batch_size, 0] = predictions
    df = pd.DataFrame(output.int().numpy(), columns=['Prediction'])
    df['Prediction'] = df['Prediction'].replace(0, -1)
    df['Id'] = df.reset_index(drop=True).index + 1
    df.to_csv(save_path, index=False)
    print("saved test set predictions to", save_path)

def print_trainable_parameters(model):
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def main():
    parser = argparse.ArgumentParser(description="Choose hyperparameters for transformer training")
    parser.add_argument('--models', nargs='+', required=True, help="List of pre-trained transformer model names")
    parser.add_argument('--seq_length', type=int, required=True, help="Sequence length")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")    
    parser.add_argument('--hidden_width', type=int, required=True, help="Number of neurons in hidden layer")
    parser.add_argument('--hidden_depth', type=int, required=True, help="Number of hidden layers")
    parser.add_argument('--freeze', action='store_true', help="Enable initial freezing of transformer parameters")
    parser.add_argument('--folder', type=str, required=False, help="Folder name, where the model and tokenizer are going to be saved.")
    parser.add_argument('--save_name', type=str, required=False, help="Model will be saved with 'save_name'.pth in the folder")
    parser.add_argument('--inference_name', type=str, required=False, help="After training, model will predict results for the test dataset and save them to 'inference_name.csv' in the save folder")
    parser.add_argument('--batch', type=int, help="batch size, default is 16")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Hyperparameters
    MODEL_NAMES = args.models
    NUM_LABELS = 2
    MAX_SEQ_LENGTH = args.seq_length
    BATCH_SIZE = args.batch if args.batch else 16
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = 1e-5
    SAVE_FOLDER = './' + args.folder if args.folder else './results'
    MODEL_SAVE = os.path.join(SAVE_FOLDER, args.save_name + '.pth' if args.save_name else 'multi_transformer.pth')
    INFERENCE_SAVE = os.path.join(SAVE_FOLDER, args.inference_name + '.csv') if args.inference_name else None

    # TensorBoard SummaryWriter
    writer = SummaryWriter()

    # Adapted from Kai's GRU implementation
    train_path_neg = os.getcwd() + "/twitter-datasets/train_neg.txt"
    train_path_pos = os.getcwd() + "/twitter-datasets/train_pos.txt"
    test_path = os.getcwd() + "/twitter-datasets/test_data.txt"

    X_train, y_train, X_val, y_val, X_test, y_test_dummy = load_data(train_path_neg, train_path_pos, test_path, val_split=0.8, frac=1.0)

    tokenizers = [AutoTokenizer.from_pretrained(model_name) for model_name in MODEL_NAMES]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Training on {n_gpus} GPUs.")

    # Initialize model, optimizer, loss
    model = MultiModelTransformer(model_names=MODEL_NAMES, num_labels=NUM_LABELS, hidden_depth=args.hidden_depth, hidden_width=args.hidden_width).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    train_dataset = preprocess_data_for_transformer(X_train, y_train, tokenizers, MAX_SEQ_LENGTH)
    val_dataset = preprocess_data_for_transformer(X_val, y_val, tokenizers, MAX_SEQ_LENGTH)
    test_dataset = preprocess_data_for_transformer(X_test, y_test_dummy, tokenizers, MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn(len(MODEL_NAMES)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn(len(MODEL_NAMES)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn(len(MODEL_NAMES)))

    os.makedirs(SAVE_FOLDER, exist_ok=True)
        
    if args.freeze:
        for transformer in model.transformers:
            for param in transformer.parameters():
                param.requires_grad = False
        print_trainable_parameters(model)

    
    best_val_loss = float('inf')  # Initialize the best validation loss to infinity

    for epoch in range(NUM_EPOCHS):
        if args.freeze and epoch == NUM_EPOCHS // 2:
            for transformer in model.transformers:
                for param in transformer.parameters():
                    param.requires_grad = True
            print(f"Unfreezing done at epoch {epoch + 1}:")
            print_trainable_parameters(model)

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

    for i, tokenizer in enumerate(tokenizers):
        tokenizer.save_pretrained(os.path.join(SAVE_FOLDER, f'tokenizer_{i}'))
    print(f"All tokenizers saved in {SAVE_FOLDER}")

    # Make predictions for test data
    predict_test(test_loader, model, INFERENCE_SAVE, device)

    writer.close()

    print("The multi-model transformer was trained in the following configuration: ")
    print(args.models, args.seq_length, args.epochs, args.hidden, args.freeze)

if __name__ == "__main__":
    main()

    # python code/transformer.py --model "bert-large-uncased" --seq_length 50 --epochs 40 --hidden 512 --freeze
    ### for tensorboard output, connect via:
    # ssh -L 6006:localhost:6006 username@remote_server_address
    ### after starting the training script:
    # tensorboard --logdir=runs --port=6006
    ### on local machine, open:
    # http://localhost:6006
