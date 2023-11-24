#use networkx to visualize the graph

import networkx as nx
from data_loader import MyDataset
import matplotlib.pyplot as plt
import os
from torch_geometric.data import Data, Dataset, DataLoader

def visualize_graph(graph):
    #graph is a torch_geometric.data.Data object
    #convert graph to networkx
    G = nx.DiGraph()
    #plot directed graph
    #add nodes
    G.add_nodes_from(range(graph.x.shape[0]))

    #add edges
    edge_index = graph.edge_index
    edge_index = edge_index.t().tolist()
    G.add_edges_from(edge_index)

    #visualize the graph
    nx.draw(G, with_labels=True,  font_color='black', font_weight='bold')

    #make title for graph using its label
    #label is 0 or 1
    label = graph.y.item()
    print(label)
    
    plt.suptitle('Graph with label ' + str(label))
    plt.show()


def main():
    cur_file_path = os.path.dirname(os.path.realpath(__file__))
    train_dataset_path = os.path.abspath(os.path.join(cur_file_path, '..', '..', '..','dataset', 'dataset_2','train'))
    dataset = MyDataset(root=train_dataset_path)
    loader = DataLoader(dataset, batch_size=1)
    for i, batch in enumerate(loader):
        visualize_graph(batch[0])
        if(i==20):
            break

if __name__ == '__main__':
    main()