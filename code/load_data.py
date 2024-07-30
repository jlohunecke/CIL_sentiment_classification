import pandas as pd


def load_data(train_path_neg, train_path_pos, test_path, val_split=0.9, frac=1):
    """
    Loads and splits the data into training, validation, and test sets.

    Parameters:
        train_path_neg (str): The file path to the negative training data.
        train_path_pos (str): The file path to the positive training data.
        test_path (str): The file path to the test data.
        val_split (float): The proportion of the data to use for training. The rest is used for validation. Default is 0.9.
        frac (float): Fraction of the data to sample. Default is 1 (use all data).

    Returns:
        tuple: A tuple containing the following:
            X_train (pd.Series): The training tweets.
            y_train (pd.Series): The training labels.
            X_val (pd.Series): The validation tweets.
            y_val (pd.Series): The validation labels.
            X_test (pd.Series): The test tweets.
            y_test_dummy (pd.Series): Dummy labels for the test data.
    """
    data_neg = pd.read_fwf(train_path_neg, header=None, names=["tweet"])
    data_neg["label"] = 0
    train_neg = data_neg[:int(val_split * len(data_neg))]
    val_neg = data_neg[int(val_split * len(data_neg)):]

    data_pos = pd.read_fwf(train_path_pos, header=None, names=["tweet"])
    data_pos["label"] = 1
    train_pos = data_pos[:int(val_split * len(data_pos))]
    val_pos = data_pos[int(val_split * len(data_pos)):]

    train = pd.concat([train_neg, train_pos], ignore_index=True).sample(frac=frac).reset_index(drop=True)
    X_train = train["tweet"].squeeze()
    y_train = train["label"].squeeze()

    val = pd.concat([val_neg, val_pos], ignore_index=True).sample(frac=frac).reset_index(drop=True)
    X_val = val["tweet"].squeeze()
    y_val = val["label"].squeeze()

    X_test = pd.read_fwf(test_path, header=None, dtype=str)[0]
    X_test = X_test.apply(lambda row: "".join(row.split(",")[1:])) # remove row indices from tweets
    y_test_dummy = pd.Series([-1] * len(X_test))
    
    return X_train, y_train, X_val, y_val, X_test, y_test_dummy
