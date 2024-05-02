import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class NaiveBayes:
    def __init__(self):
        self.pipeline = Pipeline([
            ('bow', CountVectorizer()),  # strings to token integer counts
            ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
            ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
        ])

    def fit(self, X_train, y_train):
        X_train = X_train.astype(str)
        self.pipeline.fit(X_train, y_train)
        return self.pipeline

    def evaluate(self, X_val, y_val):
        X_val = X_val.astype(str)
        y_pred = self.pipeline.predict(X_val)
        print(confusion_matrix(y_val, y_pred))
        print(classification_report(y_val, y_pred))
        print(accuracy_score(y_val, y_pred))
