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

from preprocess import load_data

from transformer_multi import CustomTextDataset, MultiModelTransformer
from transformer_multi import preprocess_data_for_transformer, custom_collate_fn, load_multi_model_and_tokenizers

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
