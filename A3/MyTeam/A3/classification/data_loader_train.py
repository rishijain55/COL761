import gzip,os
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, DataLoader

from torch_geometric.utils import one_hot

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

        last_index = len(self.graph_labels) 


        valid_indices = ~self.graph_labels.iloc[:, 0].isna()
        self.graph_labels = self.graph_labels[valid_indices]
        self.num_nodes = self.num_nodes[valid_indices]
        self.num_edges = self.num_edges[valid_indices]
        self.node_features_start = self.node_features_start[valid_indices]
        self.edge_features_start = self.edge_features_start[valid_indices]

        count0 = self.graph_labels[self.graph_labels[0] == 0].count().iloc[0]
        count1 = self.graph_labels[self.graph_labels[0] == 1].count().iloc[0]
        minclass = 0
        mincount = count0
        if count0 > count1:
            minclass = 1
            mincount = count1
        majcount = max(count0, count1)
        #oversample the minority class
        cur_ind = 0
        while mincount < majcount:
            if self.graph_labels.iloc[cur_ind, 0] == minclass:
                self.graph_labels.loc[last_index] = self.graph_labels.iloc[cur_ind]
                self.num_nodes.loc[last_index] = self.num_nodes.iloc[cur_ind]
                self.num_edges.loc[last_index] = self.num_edges.iloc[cur_ind]
                self.node_features_start.loc[last_index] = self.node_features_start.iloc[cur_ind]
                self.edge_features_start.loc[last_index] = self.edge_features_start.iloc[cur_ind]
                mincount += 1
                last_index += 1
            cur_ind += 1

    def len(self):
        return len(self.graph_labels)

    def get(self, idx):
        label = self.graph_labels.iloc[idx, 0]
        label = torch.tensor(self.graph_labels.iloc[idx, 0], dtype=torch.float32).round().long()
        #make label one dimensional
        label_shape = [1]
        labelo = torch.zeros(label_shape, dtype=torch.long)
        labelo[0] = label
        label = labelo


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
        edges = torch.tensor(self.edges.iloc[start_edge_features:end_edge_features, :].values, dtype=torch.long)

        # Construct the graph
        graph = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features, y=label)
        return graph

if __name__ == '__main__':
    cur_file_path = os.path.dirname(os.path.realpath(__file__))
    train_dataset_path = os.path.abspath(os.path.join(cur_file_path, '..', '..', '..','dataset', 'dataset_2','train'))
    print("train_dataset_path is", train_dataset_path)
    dataset = MyDataset(root=train_dataset_path)
    print(type(dataset))
    #shuffle
    # dataset = [graph for graph in dataset if graph.y.item() in [0, 1]]
    #loop over the dataset and print the labels
    print(len(dataset))
    print("graph num labels are", dataset.num_classes)
    print("graph num features are", dataset.num_node_features)

    # print("edge num features are", dataset.num_edge_features)
    # for i in range(len(dataset)):
    #     data = dataset[i]
    #     print(data.edge_index)

    # print("dataset length is", len(dataset))