import argparse
import torch
from torch_geometric.data import Data
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATv2Conv

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from torch_geometric.loader import DataLoader

import torch_geometric.transforms as T

MAX_EPOCHS = 0

def load_data(path):
    df_num_nodes = pd.read_csv(f"{path}num_nodes.csv",header = None)
    df_num_edges = pd.read_csv(f"{path}num_edges.csv",header = None)
    df_node_features = pd.read_csv(f"{path}node_features.csv",header = None)
    df_edge_features = pd.read_csv(f"{path}edge_features.csv",header = None)
    df_edge_index = pd.read_csv(f"{path}edges.csv",header = None)
    df_graph_labels = pd.read_csv(f"{path}graph_labels.csv",header = None)

    num_nodes = df_num_nodes.values
    num_graphs = len(num_nodes)
    num_edges = df_num_edges.values
    node_features = df_node_features.values
    edge_features = df_edge_features.values
    edge_indices = df_edge_index.values
    graph_labels = df_graph_labels.values

    bias = 0

    extra = len(np.where(graph_labels == 1)[0]) // len(np.where(graph_labels == 0)[0])

    if extra == 0:
        extra = len(np.where(graph_labels == 0)[0]) // len(np.where(graph_labels == 1)[0])
        bias = 1

    extra -= 1

    data_list = []
    total_nodes = 0
    total_edges = 0
    for i in range(num_graphs):

        y = 0

        if graph_labels[i][0] == 1:
            y = 1
        elif graph_labels[i][0] == 0:
            y = 0
        else:
            total_nodes += num_nodes[i][0]
            total_edges += num_edges[i][0]
            continue
        
        x = torch.tensor(node_features[total_nodes:total_nodes + num_nodes[i][0]],dtype=torch.float)
        edge_attr = torch.tensor(edge_features[total_edges:total_edges + num_edges[i][0]],dtype=torch.float)
        edge_index = torch.tensor(edge_indices[total_edges:total_edges + num_edges[i][0]].T,dtype=torch.long)
        data = Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y)
        data_list.append(data)
        if y == bias:
            for j in range(extra):
                data_list.append(data)

        total_nodes += num_nodes[i][0]
        total_edges += num_edges[i][0]
    return data_list

class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNStack, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim * 4))
        self.convs.append(self.build_conv_model(hidden_dim * 4, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

        self.dropout = 0.2
        self.num_layers = 2

    def build_conv_model(self, input_dim, hidden_dim):
        return GATv2Conv(input_dim, hidden_dim, edge_dim = 3,heads=4)

    def forward(self, data):
        x, edge_index, batch,edge_attr = data.x, data.edge_index, data.batch,data.edge_attr

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = pyg_nn.global_max_pool(x, batch)
        
        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

def train(dataset, val_dataset):

    start = time.time()
    data_size = len(dataset)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    test_loader = DataLoader(val_dataset,batch_size=1024, shuffle=True)

    # build model
    model = GNNStack(9, 64, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    training_loss = []
    validation_acc = []

    opt = optim.Adam(model.parameters(), lr=0.01)

    # train
    epoch = 0
    while time.time() - start < 60 * 25:
        total_loss = 0
        model.train()
        for batch in loader:
            #print(batch.train_mask, '----')
            batch = batch.to(device)
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        training_loss.append(total_loss)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print("Epoch {}. Training Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            validation_acc.append(test_acc)

        epoch += 1
    
    global MAX_EPOCHS
    MAX_EPOCHS = epoch
    
    return model, training_loss, validation_acc
def test(loader, model, is_validation=False):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # Load the dataset
    train_dataset = load_data(args.dataset_path)
    val_dataset = load_data(args.val_dataset_path)
    
    # Train the model
    model, training_losses, validation_acc = train(train_dataset, val_dataset)
    torch.save(model.state_dict(), f"{args.model_path}/model.pth")

    # Plot the training and validation losses
    plt.figure()
    epochs = range(0, MAX_EPOCHS)
    plt.plot(epochs, training_losses, 'r', label='Training loss',)
    epochs = range(0, MAX_EPOCHS, 10)
    plt.plot(epochs, validation_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{args.model_path}/loss.png")

if __name__=="__main__":
    main()
