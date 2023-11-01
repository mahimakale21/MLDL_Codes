import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = load_iris()
data = iris.data

# Number of clusters (K)
K = 4

# Number of iterations
iterations = 10

# Randomly initialize cluster means as data points
initial_means_indices = np.random.choice(data.shape[0], K, replace=False)
initial_means = data[initial_means_indices]

# Initialize a variable to store the final cluster means after all iterations
final_cluster_means = None

for iteration in range(iterations):
    # Create KMeans model
    kmeans = KMeans(n_clusters=K, init=initial_means, n_init=1, random_state=iteration)
    
    # Fit the model to the data
    kmeans.fit(data)
    
    # Get the final cluster means
    final_cluster_means = kmeans.cluster_centers_
    
    # Print the final cluster means for each cluster
    for cluster_num, mean in enumerate(final_cluster_means):
        print(f'Iteration {iteration + 1}, Cluster {cluster_num + 1} Mean:', mean)

    # Update the initial means for the next iteration
    initial_means = final_cluster_means

print("Final cluster means for each cluster after 10 iterations:")
for cluster_num, mean in enumerate(final_cluster_means):
    print(f'Cluster {cluster_num + 1} Mean:', mean)
