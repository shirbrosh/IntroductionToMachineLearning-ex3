import numpy as np
import abc
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from comparison import draw_points

m = [5, 10, 15, 25, 70]
k = 10000


class Classifier:

    @abc.abstractmethod
    def fit(self, X, y):
        """
        This method learns the parameters of the model and stores the trained model
        in self
        :param X: a training set, matrix withe dims d on m
        :param y: labels vector with 1 or -1 values, with dims d on 1
        """
        pass

    @abc.abstractmethod
    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        :param X: a training set, matrix withe dims d on m
        :return: a vector of predicted labels
        """
        pass

    def score(self, X, y):
        """
        Given an unlabeled test set X and the true labels y of this test set, returns
        a dictionary with the following fields: num_samples, error, accuracy, FPR, TPR,
        precision, recall.
        :param self: the model to check its score
        :param X: a training set, matrix withe dims d on m
        :param y: labels vector with 1 or -1 values, with dims d on 1
        :return: a dictionary with the following fields: num_samples, error, accuracy,
        FPR, TPR, precision, recall.
        """
        num_samples = X.shape[1]
        y_hat = self.predict(X)

        tn = np.sum((y < 0) & (y_hat < 0))
        fn = np.sum((y > 0) & (y_hat < 0))
        fp = np.sum((y < 0) & (y_hat > 0))
        tp = np.sum((y > 0) & (y_hat > 0))
        p = tp + fn
        n = tn + fp

        error_rate = (fp + fn) / (p + n)
        accuracy = (tp + tn) / (p + n)
        precision = tp / (tp + fp)
        recall = tp / p
        if n != 0:
            fpr = fp / n
        else:
            fpr = 0

        score_dict = {"num_samples": num_samples, "error": error_rate,
                      "accuracy": accuracy, "FPR": fpr, "TPR": recall,
                      "precision": precision, "recall": recall}
        return score_dict


class Perceptron(Classifier):
    def __init__(self):
        """constructor for Perceptron classifier"""
        self.w = None

    def __perceptron_algo(self, X, y):
        """
        The Perceptron classifier algorithm
        :param X: a training set, matrix withe dims d on m
        :param y: labels vector with 1 or -1 values, with dims d on 1
        :return: weights vector
        """
        w = np.zeros(X.shape[1])
        i_corrupt = np.where((X @ w).flatten() * y <= 0)[0]
        while i_corrupt.size != 0:
            x = X.T[:, i_corrupt[0]]
            w = w + y[i_corrupt[0]] * x
            i_corrupt = np.where((X @ w).flatten() * y <= 0)[0]
        return w

    def fit(self, X, y):
        """
        This method learns the parameters of the model and stores the trained model
        in self
        :param X: a training set, matrix withe dims d on m
        :param y: labels vector with 1 or -1 values, with dims d on 1
        """
        X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0).T
        w = self.__perceptron_algo(X, y)
        self.w = np.array(w)

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        :param X: a training set, matrix withe dims d on m
        :return: a vector of predicted labels
        """
        X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
        return np.sign(X.T @ self.w)


class LDA(Classifier):
    def __init__(self):
        """constructor for LDA classifier"""
        self.Py1 = None
        self.Py_minus1 = None
        self.E1 = None
        self.E_minus1 = None
        self.sigma = None

    def fit(self, X, y):
        """
        This method learns the parameters of the model and stores the trained model
        in self
        :param X: a training set, matrix withe dims d on m
        :param y: labels vector with 1 or -1 values, with dims d on 1
        """
        m = X.shape[1]
        m1 = np.sum(y > 0)
        m_minus1 = np.sum(y < 0)

        x = X.T

        # calculate the probabilities for 1 and -1
        Py1 = m1 / m
        Py_minus1 = m_minus1 / m

        # calculate the means for 1 and -1
        E1 = np.mean((x[y > 0]).T, 1)
        E_minus1 = np.mean((x[y < 0]).T, 1)

        # calculate the Covariance matrix
        sigma1 = ((x[y > 0]) - E1).T @ ((x[y > 0]) - E1)
        sigma_minus1 = ((x[y < 0]) - E_minus1).T @ ((x[y < 0]) - E_minus1)
        sigma = (1 / (m - 2)) * (sigma1 + sigma_minus1)

        self.E1 = E1
        self.E_minus1 = E_minus1
        self.Py1 = Py1
        self.Py_minus1 = Py_minus1
        self.sigma = np.array(sigma)

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        :param X: a training set, matrix withe dims d on m
        :return: a vector of predicted labels
        """
        delta1 = (X.T @ np.linalg.inv(self.sigma) @ self.E1) - 0.5 * (
                self.E1.reshape(X.shape[0]) @ np.linalg.inv(
            self.sigma) @ self.E1) + np.log(self.Py1)

        delta_minus1 = (X.T @ np.linalg.inv(self.sigma) @ self.E_minus1) - 0.5 * (
                self.E_minus1.reshape(X.shape[0]) @ np.linalg.inv(
            self.sigma) @ self.E_minus1) + np.log(
            self.Py_minus1)

        pre = np.zeros(delta1.shape)
        pre[delta1 > delta_minus1] = 1
        pre[delta1 < delta_minus1] = -1
        return pre


class SVM(Classifier):
    def __init__(self):
        """constructor for SVM classifier"""
        self.svc = SVC(C=1e10, kernel='linear')
        self.w = None

    def fit(self, X, y):
        """
        This method learns the parameters of the model and stores the trained model
        in self
        :param X: a training set, matrix withe dims d on m
        :param y: labels vector with 1 or -1 values, with dims d on 1
        """
        self.svc.fit(X.T, y)
        self.w = self.svc.coef_[0]

    def predict(self, X):
        """
         Given an unlabeled test set X, predicts the label of each sample.
         :param X: a training set, matrix withe dims d on m
         :return: a vector of predicted labels
         """
        return self.svc.predict(X.T)


class Logistic(Classifier):
    def __init__(self):
        """constructor for Logistic classifier"""
        self.logistic = LogisticRegression(solver='liblinear')
        self.w = None

    def fit(self, X, y):
        """
        This method learns the parameters of the model and stores the trained model
        in self
        :param X: a training set, matrix withe dims d on m
        :param y: labels vector with 1 or -1 values, with dims d on 1
        """
        self.logistic.fit(X, y)
        self.w = self.logistic.coef_[0]

    def predict(self, X):
        """
         Given an unlabeled test set X, predicts the label of each sample.
         :param X: a training set, matrix withe dims d on m
         :return: a vector of predicted labels
         """
        return self.logistic.predict(X)


class DecisionTree(Classifier):
    def __init__(self):
        """constructor for DecisionTree classifier"""
        self.dtc = DecisionTreeClassifier(max_depth=1)

    def fit(self, X, y):
        """
        This method learns the parameters of the model and stores the trained model
        in self
        :param X: a training set, matrix withe dims d on m
        :param y: labels vector with 1 or -1 values, with dims d on 1
        """
        self.dtc.fit(X.T, y)

    def predict(self, X):
        """
         Given an unlabeled test set X, predicts the label of each sample.
         :param X: a training set, matrix withe dims d on m
         :return: a vector of predicted labels
         """
        return self.dtc.predict(X.T)


def q_9():
    """
    This function plots the drawn points according to their label, and the hyperplane
    for- true hypothesis, hypothesis generated by the perceptron and hypothesis
    generated by SVM
    """
    for i in m:
        X, y = draw_points(i)

        X_1 = (X.T[y > 0]).T
        X_minus1 = (X.T[y < 0]).T

        # Perceptron classifier
        perceptron = Perceptron()
        perceptron.fit(X, y)
        perceptron_w = perceptron.w

        # SVM classifier
        svm = SVM()
        svm.fit(X, y)
        svm_w = svm.w
        svm_b = svm.svc.intercept_[0]

        # the plots
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Hyperplanes for ' + str(i) + " training points")
        plt.scatter(X_1[0], X_1[1], color='blue', label="true label")
        plt.scatter(X_minus1[0], X_minus1[1], color='orange', label="false label")
        x_points = np.linspace(-4, 4)
        plt.plot(x_points, ((0.3 / 0.5) * x_points + (0.1 / 0.5)),
                 label="true hypothesis hyperplane")
        plt.plot(x_points, (-(perceptron_w[1] / perceptron_w[2]) * x_points - (
                perceptron_w[0] / perceptron_w[2])),
                 label="Perceptron hypothesis hyperplane")
        plt.plot(x_points, (-(svm_w[0] / svm_w[1]) * x_points - (svm_b / svm_w[1])),
                 label="SVM hypothesis hyperplane")
        plt.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.show()


def q_10():
    """
    This function calculate and plot accuracy for 4 classifiers - Perceptron, hard SVM
    and LDA.
    """
    perceptron_acc_mean = []
    svm_acc_mean = []
    lda_acc_mean = []
    for i in m:

        perceptron_acc = []
        svm_acc = []
        lda_acc = []
        for j in range(500):

            X, y_train = draw_points(i)
            Z, y_test = draw_points(k)

            # in case the label set contains only 1's or only 0's, draw again
            while (y_train[y_train > 0].size == y_train.shape[0]) or y_train[
                y_train < 0].size == y_train.shape[0]:
                X, y_train = draw_points(i)

            # Perceptron classifier
            perceptron = Perceptron()
            perceptron.fit(X, y_train)
            perceptron_dict = perceptron.score(Z, y_test)
            perceptron_acc.append(perceptron_dict["accuracy"])

            # SVM classifier
            svm = SVM()
            svm.fit(X, y_train)
            svm_dict = svm.score(Z, y_test)
            svm_acc.append(svm_dict["accuracy"])

            # LDA classifier
            lda = LDA()
            lda.fit(X, y_train)
            lda_dict = lda.score(Z, y_test)
            lda_acc.append(lda_dict["accuracy"])

        # calculate the mean accuracy for each m
        perceptron_acc_mean.append(np.mean(perceptron_acc))
        svm_acc_mean.append(np.mean(svm_acc))
        lda_acc_mean.append(np.mean(lda_acc))

    # the plots
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Mean Accuracy Vs. num samples")
    plt.plot(m, perceptron_acc_mean, label="Perceptron")
    plt.plot(m, svm_acc_mean, label="SVM")
    plt.plot(m, lda_acc_mean, label="LDA")
    plt.legend()
    ax.set_xlabel('num samples')
    ax.set_ylabel('Mean Accuracy')
    fig.show()
