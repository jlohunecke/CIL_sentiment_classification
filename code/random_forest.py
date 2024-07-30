from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

from preprocess import preprocess, load_data, GloveEmbedder


if __name__ == "__main__":
    """
    Main script to train and evaluate a RandomForestClassifier using either GloVe or TF-IDF embeddings.
    
    Command-line arguments:
        --embedding (str): The type of embedding to use, either 'glove' or 'tfidf'. This argument is required.
    """
    # choose embedding type
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, required=True, help='type of embedding to use: glove or tfidf')
    args = parser.parse_args()
    assert args.embedding in ["glove", "tfidf"], "provided embedding method is not supported"
    # provide file paths
    train_path_neg = "twitter-datasets/train_neg.txt"
    train_path_pos = "twitter-datasets/train_pos.txt"
    test_path = "twitter-datasets/test_data.txt"
    # load and tokenize
    X_train, y_train, X_val, y_val, X_test, _ = load_data(train_path_neg, train_path_pos, test_path, val_split=0.9 ,frac=1.0)
    X_train_tokens, X_val_tokens, X_test_tokens = preprocess(X_train, X_val, X_test)
    X_train_ = [' '.join(tokens) for tokens in X_train_tokens]
    X_val_ = [' '.join(tokens) for tokens in X_val_tokens]
    X_test_ = [' '.join(tokens) for tokens in X_test_tokens]
    # create embeddings
    if args.embedding == "glove":
        embedder = GloveEmbedder()
        X_train_vectorized = embedder.transform(X_train_)
    elif args.embedding == "tfidf":
        embedder = TfidfVectorizer()
        X_train_vectorized = embedder.fit_transform(X_train_)
    X_val_vectorized = embedder.transform(X_val_)
    X_test_vectorized = embedder.transform(X_test_)
    # apply random forest
    rf = RandomForestClassifier(verbose=2)
    rf.fit(X_train_vectorized, y_train)
    y_pred = rf.predict(X_val_vectorized)

print("Validation Accuracy: ", accuracy_score(y_val, y_pred))