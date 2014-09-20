"""

"""
from sklearn.svm import LinearSVC

from data.data import trainX, trainY, testX


def naive_svm():
    """
    try a naive svm method!
    :return: numpy, test labels
    """

    # TODO 特征预处理
    # preprocess(trainX)

    clf = LinearSVC()
    clf.fit(trainX, trainY)
    y = clf.predict(testX)


if __name__ == '__main__':
    y = naive_svm()

    # write_result(y, "naive svm, first edition")
