"""
data visualization
"""
import matplotlib.pyplot as plt

plt.boxplot(trainX)
plt.savefig("./trainX_boxplot.png")
plt.show()
