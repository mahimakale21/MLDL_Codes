import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset from the Excel file
file_path = "D:/IMCC_MCA/SEM3/KRAI/Datasets/Countries_dataset2.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Display the column names
print(df.columns)

# Extract longitude and latitude as features
X = df[['Longitude', 'Latitude']]

# Choose the number of clusters (you may adjust this based on your needs)
num_clusters = 4

# Create a KMeans model
kmeans = KMeans(n_clusters=num_clusters)

# Fit the model to the data
kmeans.fit(X)

# Add cluster labels to the dataset
df['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.scatter(X['Longitude'], X['Latitude'], c=kmeans.labels_, cmap='rainbow')
plt.title('K-means Clustering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Display the cluster assignments
print(df[['Name', 'Longitude', 'Latitude', 'Cluster']])
