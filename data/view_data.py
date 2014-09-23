"""
data visualization
"""
import matplotlib.pyplot as plt
from data import trainX  # 这里pycharm会提示出错，不用管，另外commit的时候注意不要勾选optimize import

plt.boxplot(trainX)
plt.savefig("./trainX_boxplot.png")
plt.show()
