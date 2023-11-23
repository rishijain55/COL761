import numpy as np
import matplotlib.pyplot as plt

# Function to generate a dataset of random points in d-dimensional space
def generate_dataset(num_points, d):
    return np.random.uniform(0, 1, size=(num_points, d))

# Function to compute L1, L2, and Linf distances between points
def compute_distances(query_point, dataset):
    l1_distances = np.sum(np.abs(dataset - query_point), axis=1)
    l2_distances = np.sqrt(np.sum((dataset - query_point)**2, axis=1))
    linf_distances = np.max(np.abs(dataset - query_point), axis=1)
    return l1_distances, l2_distances, linf_distances

# Function to calculate the average ratio of farthest and nearest distances
def calculate_avg_ratio(farthest_distances, nearest_distances):
    return np.mean(farthest_distances / nearest_distances, axis=0)


# Main code
dimensions = [1, 2, 4]
num_points = 1000
num_queries = 100

avg_ratios_l1 = []
avg_ratios_l2 = []
avg_ratios_linf = []

for d in dimensions:
    dataset = generate_dataset(num_points, d)
    query_points = generate_dataset(num_queries, d)
    
    farthest_distances = []
    nearest_distances = []
    for query_point in query_points:
        l1, l2, linf = compute_distances(query_point, dataset)
        # Exclude the query point itself from nearest distance computation
        nearest_distance = np.array([np.min(l1[l1 != 0]), np.min(l2[l2 != 0]), np.min(linf[linf != 0])])
        farthest_distance = np.array([np.max(l1), np.max(l2), np.max(linf)])
        farthest_distances.append(farthest_distance)
        nearest_distances.append(nearest_distance)
        print(f'Query point: {query_point}, Farthest distances: {farthest_distance}, Nearest distances: {nearest_distance}')

    farthest_distances = np.array(farthest_distances)
    nearest_distances = np.array(nearest_distances)
    avg_ratio = calculate_avg_ratio(farthest_distances, nearest_distances)
    avg_ratios_l1.append(avg_ratio[0])
    avg_ratios_l2.append(avg_ratio[1])
    avg_ratios_linf.append(avg_ratio[2])
    print(f'Average ratio of farthest to nearest distances for d={d}: {avg_ratio}')

# Plot the average ratios
plt.plot(dimensions, avg_ratios_l1, label='L1 Norm')
plt.plot(dimensions, avg_ratios_l2, label='L2 Norm')
plt.plot(dimensions, avg_ratios_linf, label='L∞ Norm')
plt.xlabel('Number of Dimensions (d)')
plt.ylabel('Average Ratio of Farthest to Nearest Distances')
plt.title('Behavior of Uniformly Distributed Points in High-Dimensional Spaces')
plt.legend()
plt.savefig('p1.png')





