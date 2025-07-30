import numpy as np
from sklearn.linear_model import LogisticRegression

class SVM:
    def __init__(self, learning_rate = 0.001, lambda_param = 0.01, n_iters = 1000):
        self.lr = learning_rate
        self.lp = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.model_name = 'SVM'  


    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0
        decision_scores = []

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lp * self.w)
                else:
                    self.w -= self.lr * (2 * self.lp * self.w - x_i * y[idx])
                    self.b -= self.lr * y[idx]

        decision_scores = self.decision_function(X)
        self.platt_model = LogisticRegression()
        self.platt_model.fit(decision_scores.reshape(-1, 1), y)

    def decision_function(self, X):
        """Calcola il valore della decision function (w·x - b)"""
        return np.dot(X, self.w) - self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))
    
    def predict_proba(self, X):
        """Converte i punteggi della decision function in probabilità"""
        if self.platt_model is None:
            raise ValueError("Platt Scaling non è stato addestrato! Chiama fit() prima.")

        decision_scores = self.decision_function(X)
        probabilities = self.platt_model.predict_proba(decision_scores.reshape(-1, 1))
        return probabilities  # Restituisce la probabilità per entrambe le classi