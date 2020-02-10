from sklearn import tree
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np

from plots.plot_learning_curve import plot_learning_curve


class DecisionTree:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        self.prune_leaf_num()
        self.prune_depth()
        self.leaning_curve()

    def prune_leaf_num(self):
        param_range = [i for i in range(1, 1000, 20)]
        train_scores, test_scores = validation_curve(
            tree.DecisionTreeClassifier(), self.X_train, self.y_train, param_name="min_samples_split",
            param_range=param_range, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Prune Decision Tree: Minimum Samples Split")
        plt.xlabel(r"Number of Minimum Samples Split")
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

    def prune_depth(self):
        param_range = [i for i in range(1, 100, 10)]
        train_scores, test_scores = validation_curve(
            tree.DecisionTreeClassifier(min_samples_leaf=10), self.X_train, self.y_train, param_name="max_depth",
            param_range=param_range, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Prune Decision Tree: Max Depth")
        plt.xlabel(r"Max depth")
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
        optimized_model = tree.DecisionTreeClassifier(min_samples_split=200, max_depth=10)
        train_sizes, train_scores, valid_scores = learning_curve(optimized_model, self.X_train, self.y_train,
                                                                 train_sizes=[n for n in range(100, 850, 10)],
                                                                 cv=3)
        plot_learning_curve(train_sizes, train_scores[:, 0], valid_scores[:, 0])

