import numpy as np

COEF = np.array([0.3, -0.5])


def f(x):
    """
    This function determined the true labels for x
    :param x: the data set
    :return: the true labels for x
    """
    return np.sign(COEF @ x + 0.1)


def draw_points(m):
    """
    This function receives an integer m and returns a pair X, y where X is 2*m matrix
    where each column represents an i.i.d sample from a normal distribution with
    mean=0, and var = I2, and y from values {-1,1} is its corresponding label,
    according to f(x).
    :param m: number of samples
    """
    X = np.random.multivariate_normal([0, 0], np.eye(2), m).T
    y = f(X)
    return X, y
