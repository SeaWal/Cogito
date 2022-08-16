import numpy as np
from _core import LinearModel

class LinearRegeression(LinearModel):

    def __init__(self, lr : float = 0.001, n_iters : int = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
    

    def fit(self, X : np.ndarray, y : np.ndarray) -> None:
        n_inputs, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (2 / n_inputs) * np.dot( X.T, (y_pred - y) )
            db = (2 / n_inputs) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X : np.ndarray) -> float:
        return np.dot(X, self.weights) + self.bias


def main() -> None:
    from sklearn import datasets
    from sklearn import linear_model

    from sklearn.model_selection import train_test_split

    # create and split a dataset
    X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit the custom model
    lr = LinearRegeression(lr = 0.01)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # fit the scikit-learn version
    sklr = linear_model.LinearRegression()
    sklr.fit(X_train, y_train)
    sk_y_pred = sklr.predict(X_test)

    # mean-squared error cost function to evaluate the models
    def mse(y_true, y_predict):
        return np.mean( (y_true - y_predict)**2 )
    
    print(f"Custom model loss  =  {mse(y_test, y_pred)}")
    print(f"SKLearn model loss =  {mse(y_test, sk_y_pred)}")


if __name__ == '__main__':
    main()