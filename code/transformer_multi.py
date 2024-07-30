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

# custom data loader function
from load_data import load_data

class CustomTextDataset(Dataset):
    """Dataset class for handling multiple transformer model encodings and labels."""
    def __init__(self, encodings_list, labels):
        """
        Initializes the dataset with encodings and labels.
        
        Args:
            encodings_list (list): List of encodings from different tokenizers.
            labels (torch.Tensor): Tensor of labels.
        """
        self.encodings_list = encodings_list
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieves an item at the specified index.
        
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            dict: Dictionary containing input IDs, attention masks, and labels.
        """
        item = {f'input_ids_{i}': encodings['input_ids'][idx] for i, encodings in enumerate(self.encodings_list)}
        item.update({f'attention_mask_{i}': encodings['attention_mask'][idx] for i, encodings in enumerate(self.encodings_list)})
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        """
        Returns the length of the dataset.
        
        Returns:
            int: Number of items in the dataset.
        """
        return len(self.labels)

class MultiModelTransformer(nn.Module):
    """Neural network class for combining multiple transformer models."""
    def __init__(self, model_names, num_labels, hidden_depth, hidden_width):
        """
        Initializes the multi-model transformer.
        
        Args:
            model_names (list): List of pre-trained transformer model names.
            num_labels (int): Number of output labels.
            hidden_depth (int): Number of hidden layers in the custom head.
            hidden_width (int): Number of neurons in each hidden layer.
        """
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
        """
        Forward pass through the multi-model transformer.
        
        Args:
            input_ids_list (list): List of input IDs for each model.
            attention_mask_list (list): List of attention masks for each model.
        
        Returns:
            torch.Tensor: Output logits from the final layer.
        """
        transformer_outputs = [transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :] 
                               for transformer, input_ids, attention_mask in zip(self.transformers, input_ids_list, attention_mask_list)]
        concatenated_output = torch.cat(transformer_outputs, dim=-1)
        sequence_output = self.dropout(concatenated_output)
        logits = self.custom_layers(sequence_output)
        return logits

def preprocess_data_for_transformer(X, y, tokenizers, max_len):
    """
    Preprocesses data for input into the transformers.
    
    Args:
        X (list or pd.Series): Input texts.
        y (list or np.array): Labels corresponding to the input texts.
        tokenizers (list): List of tokenizers for the transformer models.
        max_len (int): Maximum sequence length.
    
    Returns:
        CustomTextDataset: Processed dataset ready for training.
    """
    if isinstance(X, pd.Series):
        X = X.tolist()

    encodings_list = []
    for tokenizer in tokenizers:
        encodings = tokenizer(X, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        encodings_list.append(encodings)

    labels = torch.tensor(y, dtype=torch.long)
    return CustomTextDataset(encodings_list, labels)

def custom_collate_fn(num_models):
    """
    Custom collate function for DataLoader to handle multiple transformer model inputs.

    Args:
        num_models (int): Number of transformer models.

    Returns:
        function: A collate function that processes and merges individual samples into a batch.
    """
    def collate_fn(batch):
        """
        Processes and merges individual samples into a batch.

        Args:
            batch (list): List of samples retrieved from the dataset.

        Returns:
            dict: A dictionary containing batched input IDs, attention masks, and labels.
        """
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
    """
    Trains the model for one epoch.
    
    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for the training data.
        loss_fn (function): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the training on.
    
    Returns:
        float: Average loss over the epoch.
    """
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
    """
    Evaluates the model on the validation set.
    
    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the validation data.
        loss_fn (function): Loss function.
        device (torch.device): Device to run the evaluation on.
    
    Returns:
        tuple: Average loss and accuracy over the validation set.
    """
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
    """
    Generates predictions on the test set and saves them to a CSV file compatible with Kaggle competition.
    
    Args:
        test_loader (DataLoader): DataLoader for the test data.
        model (nn.Module): The trained model.
        save_path (str): Path to save the predictions.
        device (torch.device): Device to run the prediction on.
    """
    model.eval()
    output = torch.zeros((len(test_loader.dataset)), 1)
    print("Predicting on test set...")
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
    print("Saved test set predictions to", save_path)

def load_multi_model_and_tokenizers(model_names, model_folder, directory, num_labels, hidden_depth, hidden_width, device):
    """
    Loads a MultiModelTransformer and its tokenizers from the specified directory.
    
    Args:
        model_names (list): List of pre-trained transformer model names.
        model_folder (str): Folder name where the model and tokenizers are saved.
        directory (str): Directory containing the model folder.
        num_labels (int): Number of output labels.
        hidden_depth (int): Number of hidden layers in the custom head.
        hidden_width (int): Number of neurons in each hidden layer.
        device (torch.device): Device to load the model onto.
    
    Returns:
        tuple: The loaded MultiModelTransformer and a list of tokenizers.
    """
    model_path = os.path.join(directory, model_folder)
    
    tokenizers = [AutoTokenizer.from_pretrained(os.path.join(model_path, f'tokenizer_{i}')) for i in range(len(model_names))]

    # Load the model
    model = MultiModelTransformer(model_names=model_names, num_labels=num_labels, hidden_depth=hidden_depth, hidden_width=hidden_width)
    model.load_state_dict(torch.load(os.path.join(model_path, 'multi_transformer.pth'), map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizers

def main():
    """
    Main function for training and evaluating the multi-model transformer.
    """
    parser = argparse.ArgumentParser(description="Choose hyperparameters for your custom transformer training")
    parser.add_argument('--models', nargs='+', required=True, help="List of pre-trained transformer model names coming from the Huggingface Library, for more information, see: https://huggingface.co/transformers/v4.4.2/pretrained_models.html")
    parser.add_argument('--seq_length', type=int, required=True, help="Sequence length determines the maximum length of the tokens considered when encoding the input text")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")    
    parser.add_argument('--hidden_width', type=int, required=True, help="Number of neurons per hidden layer")
    parser.add_argument('--hidden_depth', type=int, required=True, help="Number of hidden layers")
    parser.add_argument('--freeze', action='store_true', help="Enable initial freezing of transformer parameters for half of the epochs")
    parser.add_argument('--folder', type=str, required=False, help="Folder name, where the model and tokenizer are going to be saved, choose different from 'save_name' argument")
    parser.add_argument('--save_name', type=str, required=False, help="Model will be saved with '{save_name}.pth' in the folder")
    parser.add_argument('--inference_name', type=str, required=False, help="After training, model will predict results for the test dataset and save them to '{inference_name}.csv' in the save folder")
    parser.add_argument('--batch', type=int, help="Batch size, default is 16")

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

    # TensorBoard trainings- and validation curve
    writer = SummaryWriter()

    # Sentiment Twitter dataset, adapt to your personal data folder
    train_path_neg = os.getcwd() + "/twitter-datasets/train_neg_full.txt"
    train_path_pos = os.getcwd() + "/twitter-datasets/train_pos_full.txt"
    test_path = os.getcwd() + "/twitter-datasets/test_data.txt"

    X_train, y_train, X_val, y_val, X_test, y_test_dummy = load_data(train_path_neg, train_path_pos, test_path, val_split=0.8, frac=1.0)

    tokenizers = [AutoTokenizer.from_pretrained(model_name) for model_name in MODEL_NAMES]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        if args.freeze and epoch == NUM_EPOCHS // 2:
            for transformer in model.transformers:
                for param in transformer.parameters():
                    param.requires_grad = True
            print(f"Unfreezing done at epoch {epoch + 1}:")

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = eval_model(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # Save best model and do the test set prediction
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"Model saved with validation loss: {val_loss:.4f} and accuracy: {val_accuracy:.4f}")
            print("Saving test predictions for this model.")
            predict_test(test_loader, model, INFERENCE_SAVE, device)
    
    if not os.path.exists(INFERENCE_SAVE):
        predict_test(test_loader, model, INFERENCE_SAVE, device)
    
    for i, tokenizer in enumerate(tokenizers):
        tokenizer.save_pretrained(os.path.join(SAVE_FOLDER, f'tokenizer_{i}'))
    print(f"All tokenizers saved in {SAVE_FOLDER}")

    writer.close()

    print("The multi-model transformer was trained in the following configuration: ")
    print(args.models, args.seq_length, args.epochs, args.hidden_width, args.hidden_depth, args.freeze)

if __name__ == "__main__":
    main()
