import numpy as np
from numpy.linalg import inv
import math

import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


def fit(x_train, y_train, lmda):
    """ Fit Ridge Regression.

    This function  takes 3 inputs. `x_train`, `y_train`, and `lmda`.

    Parameters
    ----------
    x_train : numpy-array
        a dataset represented as a numpy-array.
    y_train : numpy-array
        a dataset represented as a numpy-array.
    lmda  : int

    Returns
    -------
    numpy-array
    Wrr: weigth vector for Ridge Regression.
    """

    a = x_train.T.dot(x_train) + lmda * np.eye(x_train.shape[1])
    b = x_train.T.dot(y_train)
    Wrr = np.linalg.inv(a).dot(b)

    return Wrr


def predict(X_test, Wrr):
    """ Predict using Ridge Regression and predicts the respective Y_test values

    This function  takes 2 inputs. `x_test`, `Wrr`.

    Parameters
    ----------
    x_test : numpy-array
        a dataset represented as a numpy-array.
    Wrr : numpy-array
        a dataset represented as a numpy-array.

    Returns
    -------
    numpy-array
    Y_predict: predicted values for the test data test.
    """

    Y_predict = X_test.dot(Wrr)

    return Y_predict
