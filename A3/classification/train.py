import argparse
from torch_geometric.data import Data
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATv2Conv

import torch
import torch.optim as optim
import time

from torch_geometric.data import DataLoader, Dataset, Data

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

        if typeInput == 'Train':
            #oversample the minority class
            last_index = len(self.num_nodes)
             
            count0 = self.graph_labels[self.graph_labels[0] == 0].count().iloc[0]
            count1 = self.graph_labels[self.graph_labels[0] == 1].count().iloc[0]
            minclass = 0
            mincount = count0
            if count0 > count1:
                minclass = 1
                mincount = count1
            majcount = max(count0, count1)
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


def train(dataset, val_dataset):
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True)

    # Build model
    edge_features_dim = dataset.num_edge_features
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes
    hid_size = (num_node_features + edge_features_dim) * 4
    model = GNNStack(num_node_features, hid_size, num_classes, edge_dim=edge_features_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Train
    training_loss = []
    validation_acc = []
    start_time = time.time()
    epoch = 0
    limit = 1800

# save model with highest validation accuracy
    best_acc = 0
    best_model = None


    while time.time() - start_time < limit:
        total_loss = 0
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, pred = model(batch)
            loss = model.loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        total_loss /= len(loader.dataset)
        training_loss.append(total_loss)

        if epoch % 5 == 0:
            test_acc = test(test_loader, model)
            print("Epoch {}. Training Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            validation_acc.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = model

        epoch += 1
    return best_model, training_loss, validation_acc


def test(loader, model, is_validation=False):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    correct = 0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        correct += pred.eq(label).sum().item()

    total = len(loader.dataset)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Training a classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    args = parser.parse_args()
    print(f"Training a classification model. Output will be saved at {args.model_path}. Dataset will be loaded from {args.dataset_path}. Validation dataset will be loaded from {args.val_dataset_path}.")
    train_dataset = MyDataset(args.dataset_path)
    train_dataset.shuffle()
    val_dataset = MyDataset(args.val_dataset_path, typeInput='Train')
    model, training_loss, validation_acc = train(train_dataset, val_dataset)
    torch.save(model.state_dict(), args.model_path)
    

if __name__=="__main__":
    main()
