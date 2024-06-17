import pandas as pd
import pdb
import os
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, load_metric
from load_data import load_data
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = 'roberta-base'
NUM_LABELS = 2
MAX_SEQ_LENGTH = 40
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
SAVE_DIRECTORY = './results'  # Set to a valid directory name
MODEL_SAVE_PATH = os.path.join(SAVE_DIRECTORY, 'model.pth')
TOKENIZER_SAVE_PATH = os.path.join(SAVE_DIRECTORY, 'tokenizer')

train_path_neg = os.getcwd() + "/twitter-datasets/train_neg.txt"
train_path_pos = os.getcwd() + "/twitter-datasets/train_pos.txt"
test_path = os.getcwd() + "/twitter-datasets/test_data.txt"

X_train, y_train, X_val, y_val, X_test = load_data(train_path_neg, train_path_pos, test_path, val_split=0.8, frac=1.0)

def preprocess_data_for_roberta(X, y, tokenizer):
    if isinstance(X, pd.Series):
        X = X.tolist()

    encodings = tokenizer(X, truncation=True, padding=True, max_length=MAX_SEQ_LENGTH)
    
    labels = torch.tensor(y)
    
    return Dataset.from_dict({'input_ids': encodings['input_ids'],
                              'attention_mask': encodings['attention_mask'],
                              'labels': labels})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

train_dataset = preprocess_data_for_roberta(X_train, y_train, tokenizer)
val_dataset = preprocess_data_for_roberta(X_val, y_val, tokenizer)

train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Training on {n_gpus} GPUs.")

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=SAVE_DIRECTORY,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    load_best_model_at_end=True,
    fp16=True,
    dataloader_num_workers=4, 
    report_to=["none"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    )

trainer.train()

model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)