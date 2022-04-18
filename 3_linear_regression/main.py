import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics


class CustomLogisticRegression:
    _estimator_type = "classifier"

    def __init__(self, eta=0.001, max_iter=1000, C=1.0, tol=1e-5, random_state=42, zero_init=False):
        """Logistic Regression classifier.

        Args:
            eta: float, default=0.001
                Learning rate.
            max_iter: int, default=1000
                Maximum number of iterations taken for the solvers to converge.
            C: float, default=1.0
                Inverse of regularization strength; must be a positive float.
                Smaller values specify stronger regularization.
            tol: float, default=1e-5
                Tolerance for stopping criteria.
            random_state: int, default=42
                Random state.
            zero_init: bool, default=False
                Zero weight initialization.
        """
        self.eta = eta
        self.max_iter = max_iter
        self.C = C
        self.tol = tol
        self.random_state = np.random.RandomState(seed=random_state)
        self.zero_init = zero_init

    def get_sigmoid(self, X, weights):
        """Compute the sigmoid value."""
        return 1 / (1 + np.exp(-np.dot(X, weights)))

    def get_loss(self, x, weights, y):
        """Calculate the loss."""
        lasso_sum = 0
        N = len(x)
        for i in range(N):
            power = (np.dot(X[i], weights) + weights[0]) * y[i]
            underlog = 1 + np.exp(-power)
            lasso_sum += np.log(underlog)

        square_sum = sum(np.square(weights))
        second_arg = 1 * square_sum / (2 * self.C)
        return lasso_sum / N + second_arg

    def fit(self, X, y):
        """Fit the model.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
                Target vector.
        """
        X_ext = np.hstack([np.ones((X.shape[0], 1)), X])  # a constant feature is included to handle intercept
        num_features = X_ext.shape[1]
        if self.zero_init:
            self.weights_ = np.zeros(num_features)
        else:
            weight_threshold = 1.0 / (2 * num_features)
            self.weights_ = self.random_state.uniform(low=-weight_threshold,
                                                      high=weight_threshold,
                                                      size=num_features)  # random weight initialization

        N = X_ext.shape[0]
        n = X_ext.shape[1]
        for t in range(self.max_iter):
            we = list()
            for j in range(X.shape[0]):
                q = 1 - 1 / (1 + np.exp(-self.weights_ @ X_ext[j] * y[j]))
                we.append(X_ext[j] * y[j] * q)
            delta = np.sum(we, axis=0) / X.shape[0] - self.weights_ / self.C
            self.weights_ -= -self.eta * delta
            # first_argument = 0
            # for i in range(N):
            #     wt_sum = self.weights_[0]
            #     for j in range(n):
            #         wt_sum += self.weights_[j] * X_ext[i][j]
            #
            #     under_exp = wt_sum * y[i]
            #     brackets = 1 - 1 / (1 + np.exp(-under_exp))
            #     first_argument += y[i] * X_ext[i] * brackets
            #
            # second_argument = self.weights_ / self.C
            # delta = second_argument - first_argument / N
            # self.weights_ -= self.eta * delta

            if np.linalg.norm(delta) < self.tol:
                break

    def predict_proba(self, X):
        """Predict positive class probabilities.

        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            y: numpy array of shape (n_samples,)
                Vector containing positive class probabilities.
        """
        X_ext = np.hstack([np.ones((X.shape[0], 1)), X])
        if hasattr(self, 'weights_'):
            return self.get_sigmoid(X_ext, self.weights_)
        else:
            raise NotFittedError("CustomLogisticRegression instance is not fitted yet")

    def predict(self, X):
        """Predict classes.

        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            y: numpy array of shape (n_samples,)
                Vector containing predicted class labels.
        """
        X_ext = np.hstack([np.ones((X.shape[0], 1)), X])
        N = len(X_ext)
        y = np.ones(N)

        for i in range(N):
            scores = self.get_sigmoid(X_ext[i], self.weights_)
            if (scores >= 0.5):
                y[i] = 1
            else:
                y[i] = -1
        return y


X, y = datasets.load_digits(n_class=10, return_X_y=True)

# _, axes = plt.subplots(nrows=3, ncols=7, figsize=(10, 5))
# for ax, image, label in zip(axes.flatten(), X, y):
#     ax.set_axis_off()
#     ax.imshow(image.reshape((8, 8)), cmap=plt.cm.gray_r if label % 2 else plt.cm.afmhot_r)
#     ax.set_title(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

y_train = (y_train % 2) * 2 - 1
y_test = (y_test % 2) * 2 - 1

lr_clf = CustomLogisticRegression(max_iter=1, zero_init=True)
lr_clf.fit(X_train, y_train)
print(lr_clf.weights_)


