from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np

from plots.plot_learning_curve import plot_learning_curve


class NeuralNetwork:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = MLPClassifier()

    def train(self):
        self.optimize_hidden_layer()
        self.optimize_alpha()
        self.leaning_curve()

    def optimize_hidden_layer(self):
        param_range = [(i, i) for i in range(3, 40, 2)]
        param_range_1d = [i for i in range(3, 40, 2)]
        train_scores, test_scores = validation_curve(
            MLPClassifier(alpha=0.00001, solver='lbfgs'), self.X_train, self.y_train, param_name="hidden_layer_sizes",
            param_range=param_range, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Optimize Neural Network: Hidden Layer Size")
        plt.xlabel(r"Number of Neurons on Each Hidden Layer")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2
        plt.semilogx(param_range_1d, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range_1d, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range_1d, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range_1d, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        plt.show()

    def optimize_alpha(self):
        param_range = [i/1000 for i in range(1, 20, 3)]
        train_scores, test_scores = validation_curve(
            MLPClassifier(hidden_layer_sizes=(10, 10)), self.X_train, self.y_train, param_name="alpha",
            param_range=param_range, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Optimize Neural Network: Alpha Regularization")
        plt.xlabel(r"Alpha")
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
        optimized_model = MLPClassifier(hidden_layer_sizes=(10, 10), alpha=0.01)
        train_sizes, train_scores, valid_scores = learning_curve(optimized_model, self.X_train, self.y_train,
                                                                 train_sizes=[n for n in range(100, 850, 10)],
                                                                 cv=3)
        plot_learning_curve(train_sizes, train_scores[:, 0], valid_scores[:, 0])