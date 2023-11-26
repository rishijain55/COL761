import argparse
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATv2Conv
import time
import networkx as nx
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Dataset, Data
import torch_geometric.transforms as T
import gzip
import os

def read_gzipped_csv(file_path):
    #include the header
    df = pd.read_csv(gzip.open(file_path), header=None)
    return df

#typeInput can be 'Train', 'Test', 'Val'


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


        if typeInput != 'Test':
            
            valid_indices = ~self.graph_labels.iloc[:, 0].isna()
            self.graph_labels = self.graph_labels[valid_indices].reset_index(drop=True)
            self.num_nodes = self.num_nodes[valid_indices].reset_index(drop=True)
            self.num_edges = self.num_edges[valid_indices].reset_index(drop=True)
            self.node_features_start = self.node_features_start[valid_indices].reset_index(drop=True)
            self.edge_features_start = self.edge_features_start[valid_indices].reset_index(drop=True)


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
            label = torch.tensor(self.graph_labels.iloc[idx, 0], dtype=torch.float32)
            graph = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features, y=label)
        else:
            graph = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features)        
    
        return graph

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
            label = torch.tensor(self.graph_labels.iloc[idx, 0], dtype=torch.float32)

            graph = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features, y=label)
        else:
            graph = Data(x=node_features, edge_index=edges.t().contiguous(), edge_attr=edge_features)        
    
        return graph
    


class GraphNeuralNetwork(nn.Module):
    heads = 4
    dropout_rate = 0.1
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dims=3):
        super(GraphNeuralNetwork, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.layer_norms.append(nn.LayerNorm(hidden_dim * self.heads))
        self.layer_norms.append(nn.LayerNorm(hidden_dim * self.heads))

        self.conv_layers.append(GATv2Conv(input_dim, hidden_dim, edge_dim = edge_dims,heads=self.heads))
        self.conv_layers.append(GATv2Conv(hidden_dim * self.heads, hidden_dim, edge_dim = edge_dims,heads=self.heads))
        self.conv_layers.append(GATv2Conv(hidden_dim * self.heads, hidden_dim, edge_dim = edge_dims,heads=self.heads))

        self.post_processing = nn.Sequential(
            nn.Linear(hidden_dim * self.heads, hidden_dim), nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

        self.num_layers = 3

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr=edge_attr)
            embedding = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if not i == self.num_layers - 1:
                x = self.layer_norms[i](x)

        x = pyg_nn.global_max_pool(x, batch)

        x = self.post_processing(x)

        return embedding, x

    def loss(self, predictions, labels):
        return F.mse_loss(predictions, labels.unsqueeze(1))



def train(dataset, val_dataset):

    start_time = time.time()
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True)

    # Build model
    num_features = dataset.num_node_features
    num_edge_features = dataset.num_edge_features
    hid_dim = (num_features + num_edge_features) * 6

    model = GraphNeuralNetwork(num_features, hid_dim, 1, edge_dims=num_edge_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005)

    training_losses = []
    validation_losses = []

    # Train
    epoch = 0
    best_loss = 10000.00
    best_model = None
    lim =1800
    while time.time() - start_time < lim:  # Train for 25 minutes
        total_loss = 0
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)
        training_losses.append(total_loss)

        if epoch % 10 == 0:
            val_loss = test(val_loader, model)
            print("Epoch {}. Training Loss: {:.4f}. Validation Loss: {:.4f}".format(
                epoch, total_loss, val_loss))
            validation_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model

        epoch += 1

    return best_model, training_losses, validation_losses

def test(loader, model):
    model.eval()

    total_loss = 0
    for data in loader:
        with torch.no_grad():
            data = data.to('cuda')
            embembedding, pred = model(data)
            label = data.y

        loss = model.loss(pred, label)
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)



def main():
    parser = argparse.ArgumentParser(description="Training a classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    args = parser.parse_args()
    print(f"Training a classification model. Output will be saved at {args.model_path}. Dataset will be loaded from {args.dataset_path}. Validation dataset will be loaded from {args.val_dataset_path}.")

    train_dataset = MyDataset(args.dataset_path)
    train_dataset.shuffle()
    #normalize
    
    
    val_dataset = MyDataset(args.val_dataset_path, typeInput='Val')
    model, training_losses, validation_losses = train(train_dataset, val_dataset)
    torch.save(model.state_dict(), args.model_path)

if __name__=="__main__":
    main()
