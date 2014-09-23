"""

"""
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import scale
from sklearn.cross_validation import cross_val_score

from data.data import trainX, trainY, testX


def naive_svm():
    """
    try a naive svm method!
    :return: numpy, test labels
    """
    preprocess(trainX)

    # find the best params
    test_results = {}
    x = [10 ** i for i in range(-3, 4, 1)]
    y = [10 ** i for i in range(-3, 4, 1)]
    xx, yy = np.meshgrid(x, y)
    for c, gamma in zip(xx.flatten(), yy.flatten()):
        clf = SVC(C=c, kernel='rbf', gamma=gamma)
        scores = cross_val_score(clf, trainX, trainY, cv=5)
        test_results[(c, gamma)] = scores
        print("mean:{0},\tc = {1},\tgamma={2}".format(scores.mean(), c, gamma))
    print(test_results)
    best_score, (bestc, bestgamma) = sorted((scores.mean(), params)
                                            for params, scores in test_results.items())[-1]
    print("best score is :", best_score)  # 0.922
    print("params : c = {}, gamma = {}".format(bestc, bestgamma))  # (c=3, gamma=0.018)

    clf = SVC(C=bestc, kernel='rbf', gamma=bestgamma)
    clf.fit(trainX, trainY)
    y = clf.predict(testX)
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
