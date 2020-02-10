import pickle
from pathlib import Path
import pandas as pd
from datetime import datetime

from plots.plot_genre_analysis import plot_genre
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


from ml_models.decision_tree import DecisionTree
from ml_models.nn import NeuralNetwork
from ml_models.gradient_boosting import GradientBoosting
from ml_models.svm import SVM
from ml_models.knn import KNN


class MovieGenrePredictor:
    def __init__(self) -> None:
        pass

    def train_model(self):
        data = pd.read_csv(Path('data_files/winequality-red.csv'), sep=';')
        X = data.iloc[:, :-1].to_numpy()
        qualities = data.iloc[:, -1].to_numpy()
        Y = self.__dicretize_quality(qualities)
        Xt = self.__feature_engineer(X)
        X_train, X_test, y_train, y_test = train_test_split(Xt, Y, test_size=0.20, random_state=42)

        # features, labels = self.__vectorize_features(path=Path("./data_files/data.csv"))
        self.__plot_features(Y)
        # features, labels = Word2VecModel().train(features, labels)
        # print(datetime.now())
        # self.__train_decision_tree(X_train, X_test, y_train, y_test)
        # print(datetime. now())
        # self.__train_nn(X_train, X_test, y_train, y_test)
        # print(datetime.now())
        # self.__train_gb(X_train, X_test, y_train, y_test)
        # print(datetime.now())
        # self.__train_svm(X_train, X_test, y_train, y_test)
        print(datetime.now())
        self.__train_knn(X_train, X_test, y_train, y_test)

    def __train_decision_tree(self, X_train, X_test, y_train, y_test):
        tree = DecisionTree(X_train, X_test, y_train, y_test)
        tree.train()

    def __train_nn(self, X_train, X_test, y_train, y_test):
        model = NeuralNetwork(X_train, X_test, y_train, y_test)
        model.train()

    def __train_gb(self, X_train, X_test, y_train, y_test):
        model = GradientBoosting(X_train, X_test, y_train, y_test)
        model.train()

    def __train_svm(self, X_train, X_test, y_train, y_test):
        model = SVM(X_train, X_test, y_train, y_test)
        model.train()

    def __train_knn(self, X_train, X_test, y_train, y_test):
        model = KNN(X_train, X_test, y_train, y_test)
        model.train()

    def __vectorize_features(self, path):
        features, labels = self.words_embedder.embed_word(path=path)
        return features, labels

    def __plot_features(self, labels):
        plot_genre(labels=labels)

    def __dicretize_quality(self, qualities):
        Y = []
        for quality in qualities:
            if quality < 5:
                Y.append('very_bad')
            elif quality < 6:
                Y.append('bad')
            elif quality < 7:
                Y.append('ordinary')
            elif quality < 8:
                Y.append('good')
            else:
                Y.append('very_good')
        return Y

    def __feature_engineer(self, X):
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        est = KBinsDiscretizer(n_bins=20, encode='onehot', strategy='uniform')
        est.fit(X)
        Xt = est.transform(X)
        return Xt


if __name__ == '__main__':
    MovieGenrePredictor().train_model()

