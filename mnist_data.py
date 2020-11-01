import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import time

m = [50, 100, 300, 500]

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_images = np.logical_or((y_train == 0), (y_train == 1))
test_images = np.logical_or((y_test == 0), (y_test == 1))
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]


def q_12():
    """
    This function draws 3 images of samples labeled with '0' and 3 images of samples
    labeled with '1'
    """
    im_lable_1 = x_train[y_train == 1][:3]
    im_lable_0 = x_train[y_train == 0][:3]

    for im in im_lable_1:
        plt.imshow(im)
        plt.show()
    for im in im_lable_0:
        plt.imshow(im)
        plt.show()


def rearrange_data(X):
    """
    This function given a data as a tensor of size m*28*28, returns a new matrix of
    size m*784 with the same data
    :param X: data to reshape
    :return: the reshaped data
    """
    m = X.shape[0]
    return X.reshape((m, 784))


def draw_points(x, y, m):
    """
    This function draw points from given data set according to uniform randomly
    selected indexes
    :param x: the data to take from
    :param y: the matching label vector to take from
    :param m: the amount of samples to draw
    :return: the selected data and labels
    """
    x_index = np.arange(x.shape[0])
    wanted_index = np.random.choice(x_index, m)
    draw_points_x = x[wanted_index]
    draw_points_y = y[wanted_index]
    return draw_points_x, draw_points_y


def q_14():
    """
    This function calculate and plot accuracy for 4 classifiers - Logistic, Decision
    Tree, soft SVM and KNeighbors.
    """
    logistic_acc_mean = []
    svm_acc_mean = []
    dt_acc_mean = []
    knn_acc_mean = []
    for i in m:

        print("training "+str(i)+" samples")
        logistic_acc = []
        svm_acc = []
        dt_acc = []
        knn_acc = []
        for j in range(50):

            print("repeat procedure number "+str(j))
            X, part_y_train = draw_points(x_train, y_train, i)

            # in case the label set contains only 1's or only 0's, draw again
            while (part_y_train[part_y_train == 0].size == part_y_train.shape[0]) or \
                    part_y_train[part_y_train == 1].size == part_y_train.shape[0]:
                X, part_y_train = draw_points(x_train, y_train, i)

            # convert the data from 3D to 2D
            X_rearrange = rearrange_data(X)
            x_test_X_rearrange = rearrange_data(x_test)

            # logistic classifier
            start_log = time.time()
            logistic = LogisticRegression(solver='liblinear')
            logistic.fit(X_rearrange, part_y_train)
            logistic_accr = logistic.score(x_test_X_rearrange, y_test)
            logistic_acc.append(logistic_accr)
            end_log = time.time()- start_log
            print("logistic classifier time for "+ str(i)+" samples is: "+str(end_log))

            # soft SVM classifier
            start_svm = time.time()
            svm = SVC(C=0.5, kernel='linear')
            svm.fit(X_rearrange, part_y_train)
            svm_acrr = svm.score(x_test_X_rearrange, y_test)
            svm_acc.append(svm_acrr)
            end_svm = time.time()- start_svm
            print("svm classifier time for " + str(i) + " samples is: " + str(
                end_svm))

            # Decision Tree classifier
            start_dt = time.time()
            dt = DecisionTreeClassifier(max_depth=1)
            dt.fit(X_rearrange, part_y_train)
            dt_acrr = dt.score(x_test_X_rearrange, y_test)
            dt_acc.append(dt_acrr)
            end_dt = time.time()- start_dt
            print("Decision Tree classifier time for " + str(i) + " samples is: " + str(
                end_dt))

            # KNeighbors classifier
            start_kn = time.time()
            knn = KNeighborsClassifier()
            knn.fit(X_rearrange, part_y_train)
            knn_acrr = knn.score(x_test_X_rearrange, y_test)
            knn_acc.append(knn_acrr)
            end_kn = time.time() - start_kn
            print("KNeighbors classifier time for " + str(i) + " samples is: " + str(
                end_kn))

        # calculate the mean accuracy for each m
        logistic_acc_mean.append(np.mean(logistic_acc))
        svm_acc_mean.append(np.mean(svm_acc))
        dt_acc_mean.append(np.mean(dt_acc))
        knn_acc_mean.append(np.mean(knn_acc))

    # the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Mean Accuracy Vs. num samples")
    plt.plot(m, logistic_acc_mean, label="Logistic")
    plt.plot(m, svm_acc_mean, label="SOFT SVM")
    plt.plot(m, dt_acc_mean, label="Decision Tree Classifier")
    plt.plot(m, knn_acc_mean, label="K Neighbors Classifier")
    plt.legend()
    ax.set_xlabel('num samples')
    ax.set_ylabel('Mean Accuracy')
    fig.show()
