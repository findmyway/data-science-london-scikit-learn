"""
show feature weights using
f_classif, Lasso, LinearSVC, LogReg, Tree
"""
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

import data

trainX = data.trainX
trainY = data.trainY
n_features = trainX.shape[-1]

# original feature distribution
plt.boxplot(trainX)
plt.title("original feature distribution")
# plt.savefig("./trainX_boxplot.png")


def plot_fig(scores, tittle):
    plt.figure()
    plt.bar(np.arange(n_features) + .6, scores)
    plt.xticks(np.arange(5, n_features + 1, 5), np.arange(5, n_features + 1, 5))
    plt.title("feature weights using " + tittle)
    plt.xlim((0, n_features + 1))
    # plt.savefig("./feature_weights(" + tittle + ").png")


scores = {}
# using p-value to evaluate features
scores['p-value'], _ = f_classif(trainX, trainY)

# using Logistic Regression to evaluate features
scaleX = scale(trainX, copy=True)
clf = LogisticRegression(penalty='l1').fit(scaleX, trainY)
scores['LogReg'] = clf.coef_[0]

# using Lasso to evaluate features
clf = Lasso(0.005).fit(scaleX, trainY)
scores['Lasso'] = clf.coef_

# using LinearSVC
clf = LinearSVC(penalty='l1', dual=False).fit(scaleX, trainY)
scores['svc'] = clf.coef_[0]

# using ensemble tree
clf = ExtraTreesClassifier().fit(trainX, trainY)
scores['tree'] = clf.feature_importances_

# plot all above together
for tittle, score in scores.items():
    plot_fig(score, tittle)

# draw all weights in one fig
# scale all weight betweent 0~1
scores_abs = {name: np.abs(score) / np.abs(score).max()
              for name, score in scores.items()}
plt.figure()
cols = 'rgbky'
for i, name in enumerate(scores_abs.keys()):
    plt.bar(np.arange(n_features) + 0.15 * i,
            scores_abs[name],
            width=0.15,
            label=name,
            color=cols[i])
plt.title("feature weights using 5 different methods")
plt.xticks(np.arange(n_features) + 0.15 * 2, np.arange(n_features) + 1)
plt.yticks(())
plt.legend(loc='upper left')
# plt.savefig("featre_weights.png")

plt.show()
