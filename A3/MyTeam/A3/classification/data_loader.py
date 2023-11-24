#load data from csv.gz file to torch_gemetric.data
#files
# edge_features.csv.gz
# node_features.csv.gz
# edges.csv.gz
# num_nodes.csv.gz
# num_edges.csv.gz
# graph_labels.csv.gz

# 1 Graph formats
# ===============

#   - The files are plaintext.
#   - There are 6 csv files (gzipped).
#   - graph_labels, num_nodes and num_edges contain as many lines as there
#     are graphs. The first line of graph_labels contains the label of
#     graph 1, first line of num_nodes contains the num of nodes in graph
#     1, and the first line of num_edges contains the num of edges in
#     graph 1.
#   - node_features: if graph 1 has n1 nodes, graph 2 has n2 nodes,
#     etc. Then first n1 lines contain node features of graph 1, next n2
#     lines contains node features of nodes of graph 2, so on. Each line
#     itself will be a vector (multiple categorical scalar values). The
#     order in which the nodes appear will be used in the edges.csv file
#     as defined below.
#   - edges: if graph 1 has e1 edges, graph 2 has e2 edges, etc. Then
#     first e1 lines contain edges (node pairs) of graph 1, first e2 lines
#     contain edges of graph 2 and so on. The node numbers are in order of
#     their appearance in the node_features file. That is if there is an
#     edge (0, 10) in graph k, then there is an edge in graph k between
#     the nodes whose node features appear at the first and the eleventh
#     (0-indexed node numbers) row in the node_features file for that
#     graph.
#   - edge_features: if graph 1 has e1 edges, graph 2 has e2 edges,
#     etc. Then first e1 lines contain attributes of graph 1, first e2
#     lines contain attributes of graph 2 and so on. Each line itself will
#     be a vector (multiple categorical scalar values).

#   - You may create `torch_geometric.data.Data' objects from these
#     graphs. Then create a `torch_geometric.data.Dataset' object, and
#     then use a `torch_geometric.loader.DataLoader' to batch these
#     graphs. Read their respective documentations.

import gzip,os
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, DataLoader

def read_gzipped_csv(file_path):
    #include the header
    df = pd.read_csv(gzip.open(file_path), header=None)
    return df


class MyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyDataset, self).__init__(root, transform, pre_transform)
        dataset_path = os.path.join(root)

        # Load your dataset files here
        self.graph_labels = read_gzipped_csv(dataset_path +  '/graph_labels.csv.gz')
        self.num_nodes = read_gzipped_csv(dataset_path +  '/num_nodes.csv.gz')
        self.num_edges = read_gzipped_csv(dataset_path + '/num_edges.csv.gz')
        self.node_features = read_gzipped_csv(dataset_path +  '/node_features.csv.gz')
        self.edges = read_gzipped_csv(dataset_path +  '/edges.csv.gz')
        self.edge_features = read_gzipped_csv(dataset_path +  '/edge_features.csv.gz')
        #define a dataframe which stroes the start of node_features for each graph
        self.node_features_start = pd.DataFrame(columns=['start'])
        self.node_features_start.loc[0] = 0
        for i in range(1, len(self.num_nodes)):
            self.node_features_start.loc[i] = self.node_features_start.loc[i-1] + self.num_nodes.iloc[i-1, 0]
        #define a dataframe which stroes the start of edge_features for each graph
        self.edge_features_start = pd.DataFrame(columns=['start'])
        self.edge_features_start.loc[0] = 0
        for i in range(1, len(self.num_edges)):
            self.edge_features_start.loc[i] = self.edge_features_start.loc[i-1] + self.num_edges.iloc[i-1, 0]


    def len(self):
        return len(self.graph_labels)

    def get(self, idx):
        label = torch.tensor(self.graph_labels.iloc[idx, 0], dtype=torch.float32)
        num_nodes = self.num_nodes.iloc[idx, 0]
        num_edges = self.num_edges.iloc[idx, 0]

        # Extract node features for the current graph
        start_node_features = self.node_features_start.iloc[idx, 0]
        end_node_features = start_node_features + num_nodes
        #datatype of node_features is numpy.ndarray
        node_features = torch.tensor(self.node_features.iloc[start_node_features:end_node_features, :].values, dtype=torch.float32)

        # Extract edge features for the current graph
        start_edge_features = self.edge_features_start.iloc[idx, 0]
        end_edge_features = start_edge_features + num_edges
        edge_features = torch.tensor(self.edge_features.iloc[start_edge_features:end_edge_features, :].values, dtype=torch.float32)

        # Extract edges for the current graph
        edges = torch.tensor(self.edges.iloc[start_edge_features:end_edge_features, :].values, dtype=torch.float32)

        # Construct the graph
        graph = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features, y=label)

        return graph

if __name__ == '__main__':
    cur_file_path = os.path.dirname(os.path.realpath(__file__))
    train_dataset_path = os.path.abspath(os.path.join(cur_file_path, '..', '..', '..','dataset', 'dataset_2','train'))
    dataset = MyDataset(root=train_dataset_path)
    loader = DataLoader(dataset, batch_size=1)


    for i, batch in enumerate(loader):
        print(batch)
        print(batch.x)
        if(i==10):
            break