from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np

from plots.plot_learning_curve import plot_learning_curve


class GradientBoosting:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        self.optimize_hidden_layer()
        self.optimize_alpha()
        self.leaning_curve()

    def optimize_hidden_layer(self):
        param_range = [i for i in range(1, 100, 5)]
        train_scores, test_scores = validation_curve(
            GradientBoostingClassifier(), self.X_train, self.y_train, param_name="n_estimators",
            param_range=param_range, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Gradient Boosting: Number of Estimators")
        plt.xlabel(r"Number of Estimators")
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

    def optimize_alpha(self):
        param_range = [i for i in range(3, 50, 3)]
        train_scores, test_scores = validation_curve(
            GradientBoostingClassifier(), self.X_train, self.y_train, param_name="max_features",
            param_range=param_range, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Gradient Boosting: Max Features")
        plt.xlabel(r"Number of Features")
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
        optimized_model = GradientBoostingClassifier(max_features=30)
        train_sizes, train_scores, valid_scores = learning_curve(optimized_model, self.X_train, self.y_train,
                                                                 train_sizes=[n for n in range(50, 850, 20)],
                                                                 cv=3)
        plot_learning_curve(train_sizes, train_scores[:, 0], valid_scores[:, 0])