import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_excel("D:\\IMCC_MCA\\SEM3\\KRAI\\Datasets\\Car_data.xlsx")

# Create the feature matrix X and the target variable y
X = data[['Volume', 'Weight']]
y = data['CO2']

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict CO2 emissions for Volume = 1300 and Weight = 3300
Volume = 1300
Weight = 3300
predicted_CO2_emission = model.predict([[Volume, Weight]])

print(f"Predicted CO2 Emission for Volume = {Volume} and Weight = {Weight} Kg: {predicted_CO2_emission[0]}")
