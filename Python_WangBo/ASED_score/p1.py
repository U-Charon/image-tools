import matplotlib.pyplot as plt
import numpy as np

dist = np.linspace(0, 10, 200)
score1 = 1 / (1 + dist)
score2 = np.exp(-dist)
l1 = plt.plot(dist, score1)
l2 = plt.plot(dist, score2)
plt.legend([l1, l2], labels=['1/(1+dist)', 'np.exp(-dist)'])
plt.xlabel(xlabel='dist')
plt.ylabel(ylabel='score')
plt.show()
