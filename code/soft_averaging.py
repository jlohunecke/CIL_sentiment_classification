import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader

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

def load_model_and_tokenizer(model_name, model_folder, directory, num_labels, hidden, device):
    model_path = os.path.join(directory, model_folder)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the model
    model = CustomTransformer(model_name=model_name, num_labels=num_labels, hidden=hidden)
    model.load_state_dict(torch.load(os.path.join(model_path, model_name + ".pth"), map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Choose hyperparameters for transformer training")
    parser.add_argument('--model_names', type=str, nargs='+', required=True, help="Name of pre-trained transformer models")    
    parser.add_argument('--model_folders', type=str, nargs='+', required=True, help="Folders of pre-trained transformer models")
    parser.add_argument('--save_directory', type=str, required=True, help="Folder that contains the model_folders subdirectories.")
    parser.add_argument('--seq_length', type=int, required=True, help="Sequence length")
    parser.add_argument('--hidden', type=int, required=True, help="Number of neurons in hidden layer")
    parser.add_argument('--inference_name', type=str, required=False, help="After training, model will predict results for the test dataset and save them to 'inference_name.csv' in the save folder")
    parser.add_argument('--batch', type=int, help="batch size, default is 16")
    
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    NUM_LABELS = 2
    MAX_SEQ_LENGTH = args.seq_length
    BATCH_SIZE = args.batch if args.batch else 16
    SAVE_FOLDER = args.save_directory if args.save_directory else './results'
    INFERENCE_SAVE = os.path.join(SAVE_FOLDER, args.inference_name + '.csv') if args.inference_name else None

    train_path_neg = os.getcwd() + "/twitter-datasets/train_neg.txt"
    train_path_pos = os.getcwd() + "/twitter-datasets/train_pos.txt"
    test_path = os.getcwd() + "/twitter-datasets/test_data.txt"

    _, _, _, _, X_test, y_test_dummy = load_data(train_path_neg, train_path_pos, test_path, val_split=0.8, frac=1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("mps")
    n_gpus = torch.cuda.device_count()
    print(f"Testing on {n_gpus} GPUs.")
    
    # Load multiple models and their tokenizers
    models = []
    tokenizers = []
    for model_name, model_folder in zip(args.model_names, args.model_folders):
        model, tokenizer = load_model_and_tokenizer(model_name, model_folder, SAVE_FOLDER, NUM_LABELS, args.hidden, device)
        models.append(model)
        tokenizers.append(tokenizer)
    print("Models and tokenizers loaded successfully.")

    # Preprocess the X_test dataset using each model's tokenizer
    test_datasets = [preprocess_data_for_transformer(X_test, y_test_dummy, tokenizer, MAX_SEQ_LENGTH) for tokenizer in tokenizers]
    test_loaders = [DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn) for test_dataset in test_datasets]

    # Get predictions and average them
    output = torch.zeros((len(test_loaders[0].dataset)), 1)
    print("predicting on test set...")
    start_idx = 0
    with torch.no_grad():
        for batches in tqdm(zip(*test_loaders), total=len(test_loaders[0])):
            batch_size = batches[0]['input_ids'].size(0)
            probabilities = torch.zeros((batch_size, NUM_LABELS)).to(device)
            
            for j, batch in enumerate(batches):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = models[j](input_ids, attention_mask)
                probabilities += outputs
            
            probabilities /= len(models)
            predictions = torch.argmax(probabilities, dim=1)
            output[start_idx:start_idx + batch_size, 0] = predictions
            start_idx += batch_size

    df = pd.DataFrame(output.int().numpy(), columns=['Prediction'])
    df['Prediction'] = df['Prediction'].replace(0, -1)
    df['Id'] = df.reset_index(drop=True).index + 1
    df.to_csv(INFERENCE_SAVE, index=False)
    print("saved test set predictions to", INFERENCE_SAVE)

if __name__ == "__main__":
    main()
