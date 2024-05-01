import pandas as pd


def load_data(train_path_neg, train_path_pos, test_path):
    train_neg = pd.read_fwf(train_path_neg, header=None, names=["tweet"])
    train_neg["label"] = 0
    train_pos = pd.read_fwf(train_path_pos, header=None, names=["tweet"])
    train_pos["label"] = 1
    train = pd.concat([train_neg, train_pos], ignore_index=True).sample(frac=1).reset_index(drop=True)
    X_train = train["tweet"].squeeze()
    y_train = train["label"].squeeze()
    X_test = pd.read_fwf(test_path, header=None, names=["tweet"]).squeeze()

    return X_train, y_train, X_test


if __name__ == '__main__':
    train_path_neg = "../data/train_neg.txt"
    train_path_pos = "../data/train_pos.txt"
    test_path = "../data/test_data.txt"
    X_train, y_train, X_test = load_data(train_path_neg, train_path_pos, test_path)
    print(X_train.head())
    print(y_train.head())
    print(X_test.head())
