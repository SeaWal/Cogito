from typing import Any
import numpy as np
from _core import LinearModel

class LogisticRegression(LinearModel):

    def __init__(self, lr : float = 0.01, n_iters : int = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters


    def fit(self, X, y) -> None:
        n_inputs, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = self._sigmoid( 
                np.dot(X, self.weights) + self.bias
            )

            dw = (2 / n_inputs) * np.dot( X.T,  (y_pred - y) )
            db = (2 / n_inputs) * np.sum( y_pred - y )

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X) -> Any:
        return  self._sigmoid( np.dot(X, self.weights) + self.bias )


    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))
