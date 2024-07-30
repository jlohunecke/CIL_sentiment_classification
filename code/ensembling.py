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
    """Dataset class for handling multiple transformer model encodings and labels."""
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

class MultiModelTransformer(nn.Module):
    """Neural network class for combining multiple transformer models."""
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
        
        hidden_layers.append(nn.Linear(hidden_width, self.num_labels))

        self.custom_layers = nn.Sequential(*hidden_layers)

    def forward(self, input_ids_list, attention_mask_list):
        transformer_outputs = [transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :] 
                               for transformer, input_ids, attention_mask in zip(self.transformers, input_ids_list, attention_mask_list)]
        concatenated_output = torch.cat(transformer_outputs, dim=-1)
        sequence_output = self.dropout(concatenated_output)
        logits = self.custom_layers(sequence_output)
        return logits

def preprocess_data_for_transformer(X, y, tokenizers, max_len):
    if isinstance(X, pd.Series):
        X = X.tolist()

    encodings_list = []
    for tokenizer in tokenizers:
        encodings = tokenizer(X, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        encodings_list.append(encodings)

    labels = torch.tensor(y, dtype=torch.long)
    return CustomTextDataset(encodings_list, labels)

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

def load_multi_model_and_tokenizers(model_names, model_folder, directory, num_labels, hidden_depth, hidden_width, device):
    model_path = os.path.join(directory, model_folder)
    
    tokenizers = [AutoTokenizer.from_pretrained(os.path.join(model_path, f'tokenizer_{i}')) for i in range(len(model_names))]

    pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    if len(pth_files) != 1:
        raise ValueError("There should be exactly one .pth file in the directory.")
    pth_file = pth_files[0]

    model = MultiModelTransformer(model_names=model_names, num_labels=num_labels, hidden_depth=hidden_depth, hidden_width=hidden_width)
    
    model.load_state_dict(torch.load(os.path.join(model_path, pth_file), map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizers

def main():
    parser = argparse.ArgumentParser(description="Choose hyperparameters for transformer training")
    parser.add_argument('--model_names', type=str, nargs='+', required=True, help="Name of pre-trained transformer models")    
    parser.add_argument('--model_folders', type=str, nargs='+', required=True, help="Folders of pre-trained transformer models")
    parser.add_argument('--save_directory', type=str, required=True, help="Folder that contains the model_folders subdirectories.")
    parser.add_argument('--seq_length', type=int, required=True, help="Sequence length")
    parser.add_argument('--hidden_depth', type=int, required=True, help="Number of hidden layers in the custom head")
    parser.add_argument('--hidden_width', type=int, required=True, help="Number of neurons per hidden layer")
    parser.add_argument('--inference_name', type=str, required=False, help="After training, model will predict results for the test dataset and save them to 'inference_name.csv' in the save folder")
    parser.add_argument('--batch', type=int, help="batch size, default is 16")
    parser.add_argument('--majority', type=bool, default=False, help="If true, we perform majority vote with the passed models.")
    
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if (len(args.model_names)%2 == 0) and args.majority:
        raise ValueError("Majority vote can only be performed with an uneven number of models.")

    NUM_LABELS = 2
    MAX_SEQ_LENGTH = args.seq_length
    BATCH_SIZE = args.batch if args.batch else 16
    SAVE_FOLDER = args.save_directory
    INFERENCE_SAVE = os.path.join(SAVE_FOLDER, args.inference_name + '.csv')

    train_path_neg = os.getcwd() + "/twitter-datasets/train_neg.txt"
    train_path_pos = os.getcwd() + "/twitter-datasets/train_pos.txt"
    test_path = os.getcwd() + "/twitter-datasets/test_data.txt"

    _, _, _, _, X_test, y_test_dummy = load_data(train_path_neg, train_path_pos, test_path, val_split=0.8, frac=1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Testing on {n_gpus} GPUs.")
    
    # Load multiple models and their tokenizers
    models = []
    tokenizers = []
    for model_name, model_folder in zip(args.model_names, args.model_folders):
        model, tokenizer = load_multi_model_and_tokenizers([model_name], args.model_folder, SAVE_FOLDER, NUM_LABELS, args.hidden_depth, args.hidden_width, device)
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
            if args.majority:
                all_predictions = torch.zeros((batch_size, len(models)), dtype=torch.long).to(device)
            else:
                probabilities = torch.zeros((batch_size, NUM_LABELS)).to(device)
        
            for j, batch in enumerate(batches):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = models[j](input_ids, attention_mask)
                probabilities += outputs
                predictions = torch.argmax(outputs, dim=1)
                all_predictions[:, j] = predictions
            
            if args.majority:
                predictions, _ = torch.mode(all_predictions, dim=1)
            else:
                probabilities /= len(models)
                predictions = torch.argmax(probabilities, dim=1)
            output[start_idx:start_idx + batch_size, 0] = predictions
            start_idx += batch_size

    df = pd.DataFrame(output.int().numpy(), columns=['Prediction'])
    df['Prediction'] = df['Prediction'].replace(0, -1)
    df['Id'] = df.reset_index(drop=True).index + 1
    df.to_csv(INFERENCE_SAVE, index=False)
    print("Saved test set predictions to", INFERENCE_SAVE)

if __name__ == "__main__":
    main()
