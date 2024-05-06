import pandas as pd


def load_data(train_path_neg, train_path_pos, test_path, val_split=0.1):
    train_neg_ = pd.read_fwf(train_path_neg, header=None, names=["tweet"])
    train_neg_["label"] = 0
    train_neg = train_neg_[:int(val_split * len(train_neg_))]
    val_neg = train_neg_[int(val_split * len(train_neg_)):]

    train_pos_ = pd.read_fwf(train_path_pos, header=None, names=["tweet"])
    train_pos_["label"] = 1
    train_pos = train_pos_[:int(val_split * len(train_pos_))]
    val_pos = train_pos_[int(val_split * len(train_pos_)):]

    train = pd.concat([train_neg, train_pos], ignore_index=True).sample(frac=1).reset_index(drop=True)
    X_train = train["tweet"].squeeze()
    y_train = train["label"].squeeze()

    val = pd.concat([val_neg, val_pos], ignore_index=True).sample(frac=1).reset_index(drop=True)
    X_val = val["tweet"].squeeze()
    y_val = val["label"].squeeze()

    X_test = pd.read_fwf(test_path, header=None, names=["tweet"]).squeeze()

    return X_train, y_train, X_val, y_val, X_test

