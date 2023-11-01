import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_excel("D:\\IMCC_MCA\\SEM3\\KRAI\\Datasets\\Stock_data.xlsx")

# Create the feature matrix X and the target variable y
X = data[['Interest_Rate', 'Unemployement_Rate']]
y = data['Stock_index_price']

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict the stock index price for Interest Rate = 3 and Unemployment Rate = 5.7
Interest_Rate = 3
Unemployment_Rate = 5.7
predicted_stock_price = model.predict([[Interest_Rate, Unemployment_Rate]])

print(f"Predicted Stock Index Price for Interest Rate = {Interest_Rate} and Unemployment Rate = {Unemployment_Rate}: {predicted_stock_price[0]}")
