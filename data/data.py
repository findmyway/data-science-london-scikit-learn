import os
from time import ctime

import numpy as np


curpath = os.path.split(os.path.realpath(__file__))[0]
trainX = np.loadtxt(curpath + "/train.csv", delimiter=",")
trainY = np.loadtxt(curpath + "/trainLabels.csv")
testX = np.loadtxt(curpath + "/test.csv", delimiter=",")


def write_result(y, comment=""):
    """
    write result into result + (time).txt,
    and also write comments into result_log.txt file
    :param y: numpy
    :param comment: str, default is "" , will not write into log file
                    (some description on result.txt,)
    :return:
    """
    result = "result---" + ctime() + ".txt"
    f = open(curpath + "/results/" + result,
             "w")
    f.write("Id,Solution\n")  # add head information to result

    for i, x in enumerate(y):
        f.write(str(i) + "," + str(int(x)) + "\n")
    f.close()

    if not comment == "":
        log = open(curpath + "/result_log.txt", "a")
        log.write(result + ":\t" + comment + "\n")
        log.close()
