##Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
# Read the house data into a data frame
df = pd.read_csv("D:\IMCC_MCA\SEM3\KRAI\Datasets\Heart.csv")
print("df = ",df)
print()

df=df.drop(columns = 'Unnamed: 0')
print("df = ",df)
print()

df['ChestPain']=df['ChestPain'].astype('category')
df['ChestPain']=df['ChestPain'].cat.codes
print("df = ",df)
print()

df['Thal']=df['Thal'].astype('category')
df['Thal']=df['Thal'].cat.codes

df['AHD']=df['AHD'].astype('category')
df['AHD']=df['AHD'].cat.codes
print("df = ",df)
print()
print("is df null? ", df.isnull().sum())
print()

df = df.dropna()
print("df = ",df)
print()

x=df.drop(columns = 'AHD')
print("x = ",x)
print()
y=df['AHD']
print("y = ",y)
print()

#Split the data into train and test data
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3, random_state =21)
print("x_train = ", x_train)
print()
print("x_test = ", x_test)
print()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
print("x_train_scaled = ",x_train_scaled)
print()

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(random_state=0).fit(x_train_scaled,y_train)

log_reg.score(x_train_scaled, y_train)
log_reg.score(x_test_scaled, y_test)

log_reg1=LogisticRegression(random_state =0,
                           C=1,
                            fit_intercept = True,
                           ).fit(x_train_scaled, y_train)
train_accuracy = log_reg.score(x_train_scaled, y_train)
test_accuracy = log_reg.score(x_test_scaled, y_test)

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
