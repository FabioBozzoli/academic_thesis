import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class AdaBoost:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []
        self.model_name = 'AdaBoost'

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X,y,sample_weight = w)
            predictions = model.predict(X)
            err = np.sum(w * (predictions != y)) / np.sum(w)
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            self.alphas.append(alpha)
            self.models.append(model)
            w = w * np.exp(-alpha * y * predictions)
            w = w / np.sum(w)
    
    def decision_function(self, X):
        F = np.zeros(X.shape[0])
        for alpha, learner in zip(self.alphas, self.models):
            F += alpha * learner.predict(X)  
        return F

    def predict(self, X):
        strong_preds = self.decision_function(X)
        return np.sign(strong_preds).astype(int)
    
    def predict_proba(self, X):
        F = self.decision_function(X)
        proba_class_1 = 1 / (1 + np.exp(-2 * F)) 
        proba_class_0 = 1 - proba_class_1 
        return np.vstack([proba_class_0, proba_class_1]).T  