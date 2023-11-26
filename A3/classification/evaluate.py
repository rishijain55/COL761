import argparse
from train import test
import torch
import os
import numpy as np
import pandas as pd
import gzip
from torch_geometric.data import Data, Dataset, DataLoader
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
import torch.nn.functional as F

def read_gzipped_csv(file_path):
    #include the header
    df = pd.read_csv(gzip.open(file_path), header=None)
    return df


class MyDataset(Dataset):
    types_inputs = ['Train', 'Test', 'Val']
    def __init__(self, root, typeInput='Train', transform=None, pre_transform=None):
        super(MyDataset, self).__init__(root, transform, pre_transform)
        dataset_path = os.path.join(root)

        #check if type is in types
        if typeInput not in self.types_inputs:
            raise ValueError('typeInput must be one of %r.' % self.types_inputs)

        self.typeInput = typeInput

        # Load your dataset files here
        if typeInput != 'Test':
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


        if typeInput == 'Train':

            valid_indices = ~self.graph_labels.iloc[:, 0].isna()
            self.graph_labels = self.graph_labels[valid_indices].reset_index(drop=True)
            self.num_nodes = self.num_nodes[valid_indices].reset_index(drop=True)
            self.num_edges = self.num_edges[valid_indices].reset_index(drop=True)
            self.node_features_start = self.node_features_start[valid_indices].reset_index(drop=True)
            self.edge_features_start = self.edge_features_start[valid_indices].reset_index(drop=True)
            last_index = len(self.num_nodes)
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
        return len(self.num_nodes)

    def get(self, idx):
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
        if self.typeInput != 'Test':
            label = self.graph_labels.iloc[idx, 0]
            label = torch.tensor(self.graph_labels.iloc[idx, 0], dtype=torch.float32).round().long()
            #make label one dimensional
            label_shape = [1]
            labelo = torch.zeros(label_shape, dtype=torch.long)
            labelo[0] = label
            label = labelo
            graph = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features, y=label)
        else:
            graph = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features)        
    
        return graph

def tocsv(y_arr, *, task):
    assert task in ["classification", "regression"], f"task must be either \"classification\" or \"regression\". Found: {task}"
    assert isinstance(y_arr, np.ndarray), f"y_arr must be a numpy array, found: {type(y_arr)}"
    assert len(y_arr.squeeze().shape) == 1, f"y_arr must be a vector. shape found: {y_arr.shape}"
    assert not os.path.isfile(f"y_{task}.csv"), f"File already exists. Ensure you are not calling this function multiple times (e.g. when looping over batches). Read the docstring. Found: y_{task}.csv"
    y_arr = y_arr.squeeze()
    df = pd.DataFrame(y_arr)
    df.to_csv(f"y_{task}.csv", index=False, header=False)

def return_pred(loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    pred_list = []
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            _, pred = model(data)
            pred = pred.argmax(dim=1)
            pred_list.append(pred)
    pred_list = torch.cat(pred_list).cpu().numpy()
    return pred_list

class GNNStack(nn.Module):

    negative_slope = 0.2
    heads = 4
    dropout = 0.1
    
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim=3):
        super(GNNStack, self).__init__()
        print("GNNStack init with head %d, hidden_dim %d, output_dim %d, dropout %f" % (self.heads, hidden_dim, output_dim, self.dropout))
        # Convolutional layers
        self.convs = nn.ModuleList([
            GATv2Conv(input_dim, hidden_dim, heads=self.heads,  edge_dim=edge_dim),
            GATv2Conv(hidden_dim * self.heads, hidden_dim, heads=self.heads,  edge_dim=edge_dim),
        ])

        # Layer normalization
        self.lns = nn.ModuleList([nn.LayerNorm(hidden_dim * self.heads)])

        # Post-message-passing layers
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim * self.heads, hidden_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.num_layers = len(self.convs)



    def forward(self, data):
        x, edge_index, batch,edge_attr = data.x, data.edge_index, data.batch,data.edge_attr

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i < self.num_layers - 1:
                x = self.lns[i](x)

        x = pyg_nn.global_max_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)




def main():
    parser = argparse.ArgumentParser(description="Evaluating the classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()
    print(f"Evaluating the classification model. Model will be loaded from {args.model_path}. Test dataset will be loaded from {args.dataset_path}.")
    dataset = MyDataset(root=args.dataset_path, typeInput="Test")

    num_classes = 2
    num_node_features = dataset.num_node_features
    num_edge_features = dataset.num_edge_features
    hidden_layers = (num_node_features+num_edge_features)*4

    print("num_node_features", num_node_features, "num_edge_features", num_edge_features, "hidden_layers", hidden_layers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNStack(num_node_features, hidden_layers, num_classes, edge_dim=num_edge_features).to(device)
    path_mod = os.path.join(args.model_path)
    model.load_state_dict(torch.load(path_mod))
    model.eval()

    numpy_ys = return_pred(DataLoader(dataset, batch_size=1024), model)
    tocsv(numpy_ys, task="classification")

if __name__=="__main__":
    main()
