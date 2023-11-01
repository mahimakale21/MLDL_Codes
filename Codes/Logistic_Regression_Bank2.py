import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset from the Excel file
file_path = "D:/IMCC_MCA/SEM3/KRAI/Datasets/Bank data set 2.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Display the column names
print(df.columns)

# Extract the features and target variable
X = df[['Duration ']]
y = df['Subscription']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
