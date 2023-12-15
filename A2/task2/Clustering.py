import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Load your dataset from the .dat file
data_file = sys.argv[1]
Dim = int(sys.argv[2])
plot_file = sys.argv[3]
data = np.genfromtxt(data_file)
X = data

# Define a range of K values to test
k_values = range(1, 16)  # You can adjust the range as needed

# Initialize an empty list to store the sum of squared distances for each K
inertia = []

# Perform K-means clustering for different values of K
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow plot
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o', linestyle='-', color='b')
plt.title('Elbow Plot for K-Means Clustering')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances')
plt.xticks(k_values)
plt.grid(True)
plt.savefig(f'{plot_file}')
