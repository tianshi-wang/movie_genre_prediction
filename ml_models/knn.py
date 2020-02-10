from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np

from plots.plot_learning_curve import plot_learning_curve


class KNN:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        self.optimize_C()
        self.leaning_curve()

    def optimize_C(self):
        param_range = [i for i in range(1, 200, 2)]
        train_scores, test_scores = validation_curve(
            KNeighborsClassifier(), self.X_train, self.y_train, param_name="n_neighbors",
            param_range=param_range, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("K Nearest Neighbors: Number of Neighbors")
        plt.xlabel(r"Number of Neighbors")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        plt.show()

    def leaning_curve(self,):
        optimized_model = KNeighborsClassifier(n_neighbors=40)
        train_sizes, train_scores, valid_scores = learning_curve(optimized_model, self.X_train, self.y_train,
                                                                 train_sizes=[n for n in range(50, 850, 20)],
                                                                 cv=3)
        plot_learning_curve(train_sizes, train_scores[:, 0], valid_scores[:, 0])