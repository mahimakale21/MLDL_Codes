import numpy as np
x = np.arange(10, 20)
x
#array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
y
#array([ 2,  1,  4,  5,  8, 12, 18, 25, 96, 48])

r = np.corrcoef(x, y)
r
r[0, 1]

import numpy as np
import scipy.stats
x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
scipy.stats.pearsonr(x, y)    # Pearson's r
scipy.stats.spearmanr(x, y)   # Spearman's rho 
scipy.stats.pearsonr(x, y)[0]    # Pearson's r
scipy.stats.spearmanr(x, y)[0]   # Spearman's rho

import pandas as pd
movies = pd.read_csv("D:\IMCC_MCA\SEM3\KRAI\Datasets\MoviesOnStreamingPlatforms.csv")
movies.corr()

correlations = movies.corr()
print(correlations)
print(correlations["Year"])

import numpy as np
np.random.seed(1)
# 1000 random integers between 0 and 50
x = np.random.randint(0, 50, 1000)
# Positive Correlation with some noise
y = x + np.random.normal(0, 10, 1000)
np.corrcoef(x, y)

#always remember your magic function(%) if using Jupyter
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(correlations)
plt.show()
