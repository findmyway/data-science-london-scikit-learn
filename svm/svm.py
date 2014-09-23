"""

"""
from pprint import pprint

from sklearn.svm import SVC
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale

from data.data import trainX, trainY, testX


def naive_svm():
    """
    try a naive svm method!
    :return: numpy, test labels
    """
    preprocess(trainX)

    # grid search
    params = {'kernel': ['rbf'],
              'C': [0.5, 1, 2, 3, 4, 5],
              'gamma': np.arange(0.01, 0.02, 0.001)}
    svc = SVC()
    clf = GridSearchCV(svc, params, cv=5, n_jobs=-1)
    clf.fit(trainX, trainY)
    pprint(clf.grid_scores_)
    print(clf.best_params_, clf.best_score_)

    # create result
    best_clf = clf.best_estimator_
    best_clf.fit(trainX, trainY)
    y = best_clf.predict(testX)
    return y


def preprocess(x):
    """
    do some preprocessing on raw data
    :param x: numpy array
    :return:
    """
    # scale to standard distribution
    scale(x)


if __name__ == '__main__':
    results = naive_svm()
