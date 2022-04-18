import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sklearn


class CustomKNeighborsClassifier:
    _estimator_type = "classifier"

    def __init__(self, n_neighbors=5, weights='distance', eps=1e-9):
        """K-Nearest Neighbors classifier.

        Args:
            n_neighbors: int, default=5
                Number of neighbors to use by default for :meth:`kneighbors` queries.
            weights : {'uniform', 'distance'} or callable, default='uniform'
                Weight function used in prediction.  Possible values:
                - 'uniform' : uniform weights.  All points in each neighborhood
                  are weighted equally.
                - 'distance' : weight points by the inverse of their distance.
                  in this case, closer neighbors of a query point will have a
                  greater influence than neighbors which are further away.
            eps : float, default=1e-5
                Epsilon to prevent division by 0
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.eps = eps

    def get_pairwise_distances(self, X, Y):
        """
        Returnes matrix of the pairwise distances between the rows from both X and Y.
        Args:
            X: numpy array of shape (n_samples, n_features)
            Y: numpy array of shape (k_samples, n_features)
        Returns:
            P: numpy array of shape (n_samples, k_samples)
                Matrix in which (i, j) value is the distance
                between i'th row from the X and j'th row from the Y.
        """

        n_samples = X.shape[0]
        k_samples = Y.shape[0]

        x = (X * X).sum(axis=1).reshape(n_samples, 1) * np.ones(shape=(1, k_samples))
        y = (Y * Y).sum(axis=1) * np.ones(shape=(n_samples, 1))
        return x + y - 2 * X.dot(Y.T)

        # if squared == False:

        # return np.linalg.norm(X, Y)
        # n_samples = X.shape[0]
        # n_features = X.shape[1]
        # k_samples = Y.shape[0]
        #
        # x = (X * X).sum(axis=1).reshape((n_samples, 1)) * np.ones(shape=(1, k_samples))
        # y = (Y * Y).sum(axis=1) * np.ones(shape=(k_samples, 1))
        # return x + y - 2 * X.dot(Y.T)

    def get_class_weights(self, y, weights):
        """
        Returns a vector with sum of weights for each class
        Args:
            y: numpy array of shape (n_samles,)
            weights: numpy array of shape (n_samples,)
                The weights of the corresponding points of y.
        Returns:
            p: numpy array of shape (n_classes)
                Array where the value at the i-th position
                corresponds to the weight of the i-th class.
        """
        n_classes = len(self.classes_)
        n_samples = len(y)
        classes_weights = np.zeros(n_classes)

        res = dict(zip(self.classes_, classes_weights))

        for i in range(n_samples):
            res[y[i]] += weights[i]

        print(res.values())
        return [2, 4, 0]

    def fit(self, X, y):
        """Fit the model.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
                Target vector.
        """
        self.points = X
        self.y = y
        self.classes_ = np.unique(y)

    def predict_proba(self, X):
        """Predict positive class probabilities.

        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            y: numpy array of shape (n_samples, n_classes)
                Vector containing positive class probabilities.
        """
        if hasattr(self, 'points'):
            P = self.get_pairwise_distances(X, self.points)
            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            y = np.zeros((n_samples, n_classes))
            weights_of_points = np.ones(P.shape)
            if self.weights == 'distance':
                weights_of_points = 1 / (P + self.eps)

            for i in range(n_samples):
                min_indexes = P[i].argsort()[:self.n_neighbors]
                for index in min_indexes:
                    class_ = self.y[index]
                    y[i][class_] += weights_of_points[i][index]

                y[i] = y[i] / sum(y[i])
            return y

        else:
            raise NotFittedError("CustomKNeighborsClassifier instance is not fitted yet")

    def predict(self, X):
        """Predict classes.

        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            y: numpy array of shape (n_samples,)
                Vector containing predicted class labels.
        """
        y = []
        proba = self.predict_proba(X)
        for row in proba:
            max_el = max(row)
            a = np.where(row == max_el)[0][0]
            y.append(a)
        return y


def fit_evaluate(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    # disp = metrics.plot_confusion_matrix(clf, X_test, y_test, normalize='true')
    # disp.figure_.suptitle("Confusion Matrix")
    # plt.show()

    return metrics.accuracy_score(y_pred=clf.predict(X_train), y_true=y_train), \
           metrics.accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)


model = CustomKNeighborsClassifier(n_neighbors=5, weights='distance')
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

X, y = datasets.load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

model.fit(X_train, y_train)
knn.fit(X_train, y_train)

# y_custom = model.predict_proba(X_test)
# y_real = knn.predict_proba(X_test)

train_acc, test_acc = fit_evaluate(model, X_train, y_train, X_test, y_test)
print(train_acc, test_acc)
# print(y_custom)
# print("-----------")
# print(y_real)
