##Regression

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#in case of jupyter notebook:
# %matplotlib inline

# Read the house data into a data frame
df = pd.read_csv("D:\IMCC_MCA\SEM3\KRAI\Datasets\kc_house_data.csv")

# Display the first five observations
df.head()

# Describe the dataset
df.describe().round(2)

# Plot Pearson correlation matrix
fig_1 = plt.figure(figsize=(12, 10))
new_correlations = df.corr()
sns.heatmap(new_correlations, annot=True, cmap='Greens', annot_kws={'size': 8})
plt.title('Pearson Correlation Matrix')
plt.show()

# Determine the highest intercorrelations
highly_correlated_features = new_correlations[new_correlations > 0.75]
highly_correlated_features.fillna('-') 

# Remove features which are highly correlated with "sqft_living"
df = df.drop(['sqft_above', 'sqft_living15'], axis=1)

# Update features and store their length
features = df.iloc[:, 1:].columns.tolist()
len_of_features = len(features)
len_of_features

# There are many outliers in in price
df.boxplot(column='price',notch=True,grid=False,figsize=(6,4))
plt.show()

#Price is left skewed
sns.distplot(df.price)
plt.show()

#Sdft living is also left skewed
sns.distplot(df.sqft_living)
plt.show()

#Bivariate analysis
df1=df.drop(['id','zipcode','lat','long','date'],axis=1)

#To know on which all attributes does the price depend on 
sns.pairplot(df1,diag_kind='kde')

#There is a positive relation between price and grade
df1.plot(kind='scatter', x='grade',y='price',figsize=(5,5))

#There is a positive relation between price and sqft_;living ass well
df1.plot(kind='scatter', x='sqft_living',y='price',color='orange',figsize=(5,5))

#There is not much relation between price and waterfront
df1.plot(kind='scatter', x='waterfront',y='price',figsize=(5,5))

#There is not much relation between price and view
df1.plot(kind='scatter', x='view',y='price',figsize=(5,5))
