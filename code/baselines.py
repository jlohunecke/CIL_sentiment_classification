from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler

class NaiveBayes:
    def __init__(self):
        self.model = Pipeline([('Normalizing',MinMaxScaler()),('MultinomialNB',MultinomialNB())])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        print(confusion_matrix(y_val, y_pred))
        print(classification_report(y_val, y_pred))
        print(accuracy_score(y_val, y_pred))
