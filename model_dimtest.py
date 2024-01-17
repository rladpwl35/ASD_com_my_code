import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from loader import BioDataset
from dataloader import DataLoaderFinetune
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        input_layer (bool): whethe the GIN conv is applied to input layer or not. (Input node labels are uniform...)

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add", input_layer = False):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            # self.input_node_embeddings = torch.nn.Embedding(232, emb_dim)
            # torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)
            self.linear1 = torch.nn.Linear(232,emb_dim)
            torch.nn.init.xavier_uniform_(self.linear1.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        ##add features corresponding to self-loop edges.
        #self_loop_attr = torch.zeros(x.size(0), 9)
        #self_loop_attr[:,7] = 1 # attribute for self-loop edge
        #self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        #edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        #edge_embeddings = self.edge_encoder(edge_attr)
        ###여기서 왜 [232,232] --> [232,232,300] 되는거지......
        if self.input_layer:
            #x = self.input_node_embeddings(x.to(torch.int64))
            x = self.linear1(x.to(torch.float32))
        # return self. pagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

        return self.propagate(edge_index, x=x)
        #return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
        #return self.propagate(edge_index[0], x=x, edge_attr=edge_attr)

    def message(self, x_j):
        #return torch.cat([x_j, edge_attr], dim = 1)
        return x_j

    def update(self, aggr_out):
        #return aggr_out 
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add", input_layer = False):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            #self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            #torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)
            self.linear1 = torch.nn.Linear(232,emb_dim)
            torch.nn.init.xavier_uniform_(self.linear1.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
        
       # #add features corresponding to self-loop edges.
        #self_loop_attr = torch.zeros(x.size(0), 9)
        #self_loop_attr[:,7] = 1 # attribute for self-loop edge
        #self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        #edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        #edge_embeddings = self.edge_encoder(edge_attr)
    
        if self.input_layer:
            #x = self.input_node_embeddings(x.to(torch.int64).view(-1,))
            x = self.linear1(x.to(torch.float32))
    
        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        # print("edge_index[0] : ", edge_index[0])
        # print("edge_index[0].shape :", edge_index[0].shape)
        # print("edge_index[0].dim()", edge_index[0].dim())    
        # print("edge_index : ", edge_index)
        # print("edge_index.shape : ", edge_index.shape)
        # print("edge_index.dim()" , edge_index.dim())
        
        #return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm = norm)
        return self.propagate( edge_index, x=x, edge_attr=edge_attr, norm = norm)
    
    def message(self, x_j, edge_attr, norm):
        #return norm.view(-1, 1) * (x_j + edge_attr)
        return norm.view(-1, 1) * x_j

class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add", input_layer = False):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, heads * emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))

       # #add features corresponding to self-loop edges.
       # self_loop_attr = torch.zeros(x.size(0), 9)
       # self_loop_attr[:,7] = 1 # attribute for self-loop edge
       # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
       # edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

       # edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        #return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index, x_i, x_j, edge_attr):
        #edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        #x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean", input_layer = False):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        
        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer: 
            # self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            # torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)
            self.linear1 = torch.nn.Linear(232,emb_dim)
            torch.nn.init.xavier_uniform_(self.linear1.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        #self_loop_attr = torch.zeros(x.size(0), 9)
        #self_loop_attr[:,7] = 1 # attribute for self-loop edge
        #self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        #edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        #edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            # x = self.input_node_embeddings(x.to(torch.int64).view(-1,))
            x = self.linear1(x.to(torch.float32))
            

        x = self.linear(x)  
        return self.propagate(edge_index, x=x, aggregate=self.aggr)

    def message(self, x_j):
        #return x_j + edge_attr
        return x_j
        
    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)


class GNN(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gat, graphsage, gcn
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False

            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add", input_layer = input_layer))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, input_layer = input_layer))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, input_layer = input_layer))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, input_layer = input_layer))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr):
        h_list = [x]
        #print("h_list : ", h_list)
        for layer in range(self.num_layer):
            #print("first_edge_index : ", edge_index)
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            #print(f"{layer} layer whiat is h : ", h, h.shape)
            if layer == self.num_layer - 1:
                #remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)[0]
        #print('node representation', node_representation.shape)
        return node_representation

def visualize(emb: torch.tensor, node_type: torch.tensor):
    z = TSNE(n_components=2).fit_transform(emb.detach().cpu())
    z = np.array(z)
    plt.close()
    plt.figure(figsize=(10,10))
    plt.scatter(z[:, 0], z[:, 1], s=70, c=node_type, cmap="Set2")
    plt.show()

class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
        self.linear = torch.nn.Linear(self.emb_dim, 100)
        self.graph_pred_linear = torch.nn.Linear(100, self.num_tasks)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, data, do_visualize=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        node_representation = self.gnn(x, edge_index, edge_attr)
        
        pooled = self.pool(node_representation, batch)
        
        #graph_rep = torch.cat([pooled, center_node_rep], dim = 1)
        graph_rep = self.linear(pooled)
        
        if do_visualize:
            visualize(graph_rep, data.y)      
  
        return self.graph_pred_linear(graph_rep)


if __name__ == "__main__":
    pass



