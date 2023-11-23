import pandas as pd
import gzip
import os

def read_gzipped_csv(file_path):
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f)
    return df

def load_graph_data():
    current_path = os.path.dirname(os.path.abspath(__file__))
    #move two folder back and then to dataset
    dataset_path = os.path.abspath(os.path.join(current_path, '..', '..', '..', 'dataset', 'dataset_2', 'train'))
    print(dataset_path)
    # Read graph labels
    graph_labels = read_gzipped_csv(dataset_path + '/graph_labels.csv.gz')

    # Read num_nodes and num_edges
    num_nodes = read_gzipped_csv(dataset_path + '/num_nodes.csv.gz')
    num_edges = read_gzipped_csv(dataset_path + '/num_edges.csv.gz')

    # Read node features
    node_features = read_gzipped_csv(dataset_path + '/node_features.csv.gz')

    # Read edges
    edges = read_gzipped_csv(dataset_path + '/edges.csv.gz')

    # Read edge features
    edge_features = read_gzipped_csv(dataset_path + '/edge_features.csv.gz')

    # Store node features for each graph in a list
    node_features_list = []
    start_idx = 0

    for idx, num_node in enumerate(num_nodes):
        num_node = int(num_node[0])
        end_idx = start_idx + num_node
        node_features_list.append(node_features.iloc[start_idx:end_idx])
        start_idx = end_idx

    # Store edge features for each graph in a list
    edge_features_list = []
    start_idx = 0

    for idx, num_edge in enumerate(num_edges):
        num_edge = int(num_edge[0])
        end_idx = start_idx + num_edge
        edge_features_list.append(edge_features.iloc[start_idx:end_idx])
        start_idx = end_idx

    edge_list = []
    start_idx = 0

    for idx, num_edge in enumerate(num_edges):
        num_edge = int(num_edge[0])
        end_idx = start_idx + num_edge
        edge_list.append(edges.iloc[start_idx:end_idx])
        start_idx = end_idx

    return graph_labels, num_nodes, num_edges, node_features_list, edges, edge_features_list

# Example usage
def main():
    dataset_path = '/home/slowblow/sem7/col761/ass-git/A3/dataset/dataset_2/train'
    graph_labels, num_nodes, num_edges, node_features_list, edge_list, edge_features_list = load_graph_data()

    # Print some information for verification
    print("Graph Labels:")
    print(graph_labels.head())

    print("\nNum Nodes:")
    print(num_nodes.head())

    print("\nNum Edges:")
    print(num_edges.head())

    # Print node features for the first graph
    print("\nNode Features for Graph 1:")
    print(node_features_list[0].head())

    # Print edge features for the first graph
    print("\nEdge Features for Graph 1:")
    print(edge_features_list[0].head())

if __name__=="__main__":
    main()
