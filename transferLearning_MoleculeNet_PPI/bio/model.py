import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.conv import GATConv
import torch.nn.functional as F
from loader import BioDataset
from dataloader import DataLoaderFinetune
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax




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
            self.linear1 = torch.nn.Linear(116,emb_dim)
            torch.nn.init.xavier_uniform_(self.linear1.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        #edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
        self_loop = 'no'    
        if self_loop =='no':
            self_loop_mask = edge_index[0] == edge_index[1]
            edge_index = edge_index[:, ~self_loop_mask]

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
            self.linear1 = torch.nn.Linear(116,emb_dim)
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
        #edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
        self_loop = 'no'    
        if self_loop =='no':
            self_loop_mask = edge_index[0] == edge_index[1]
            edge_index = edge_index[:, ~self_loop_mask]
        
        
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
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 2, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        
        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            #self.bias = Parameter(torch.Tensor(heads * out_channels))
            self.bias = Parameter(torch.Tensor(out_channels))
            self.lin = Linear(heads * out_channels, out_channels, False)
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels
        
        
        self_loop = 'no'    
        if self_loop =='no':
            self_loop_mask = edge_index[0] == edge_index[1]
            edge_index = edge_index[:, ~self_loop_mask]
        
        
        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
            out = self.lin(out)
            
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



# class GAT_Conv(MessagePassing):
#     def __init__(self, emb_dim, heads=3, negative_slope=0.2, aggr = "add", input_layer = False):
#         super(GAT_Conv, self).__init__()

#         self.aggr = aggr

#         self.emb_dim = emb_dim
#         self.heads = heads
#         self.negative_slope = negative_slope

#         self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
#         self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

#         self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

#         ### Mapping 0/1 edge features to embedding
#         self.edge_encoder = torch.nn.Linear(9, heads * emb_dim)

#         ### Mapping uniform input features to embedding.
#         self.input_layer = input_layer
#         if self.input_layer:
#             self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
#             torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.att)
#         zeros(self.bias)

#     def forward(self, x, edge_index, edge_attr):
#         #add self loops in the edge space
#         edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        
#         self_loop = 'no'    
#         if self_loop =='no':
#             self_loop_mask = edge_index[0] == edge_index[1]
#             edge_index = edge_index[:, ~self_loop_mask]
#         edge_index = edge_index.to(torch.long)
        
#         print(edge_index.dtype)
#        # #add features corresponding to self-loop edges.
#        # self_loop_attr = torch.zeros(x.size(0), 9)
#        # self_loop_attr[:,7] = 1 # attribute for self-loop edge
#        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
#        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

#        # edge_embeddings = self.edge_encoder(edge_attr)

#         if self.input_layer:
#             x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

#         x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
#         #return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
#         return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

#     def message(self, edge_index, x_i, x_j, edge_attr):
#         #edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
#         #x_j += edge_attr

#         alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         alpha = softmax(alpha, edge_index[0])

#         return x_j * alpha.view(-1, self.heads, 1)

#     def update(self, aggr_out):
#         aggr_out = aggr_out.mean(dim=1)
#         aggr_out = aggr_out + self.bias

#         return aggr_out


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
            self.linear1 = torch.nn.Linear(200,emb_dim)
            torch.nn.init.xavier_uniform_(self.linear1.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        
        ##edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
        self_loop = 'no'    
        if self_loop =='no':
            self_loop_mask = edge_index[0] == edge_index[1]
            edge_index_self_loop = edge_index[:, self_loop_mask]
            edge_index = edge_index[:, ~self_loop_mask]
        
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

        #if self.num_layer < 2:
        #    raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        
        if gnn_type == 'gat':
            if num_layer == 1:
                self.gnns.append(GATConv(116, emb_dim))
            else:
                self.gnns.append(GATConv(116, emb_dim))
                
                for layer in range(1, num_layer):
                    self.gnns.append(GATConv(emb_dim, emb_dim))
        else:
            for layer in range(num_layer):
                if layer == 0:
                    input_layer = True
                else:
                    input_layer = False

                if gnn_type == "gin":
                    self.gnns.append(GINConv(emb_dim, aggr = "add", input_layer = input_layer))
                elif gnn_type == "gcn":
                    self.gnns.append(GCNConv(emb_dim, input_layer = input_layer))
                # elif gnn_type == "gat":
                #     self.gnns.append(GATConv(emb_dim, input_layer = input_layer))
                elif gnn_type == "graphsage":
                    self.gnns.append(GraphSAGEConv(emb_dim, input_layer = input_layer))

    def forward(self, x, edge_index, edge_attr):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
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
        self.graph_pooling = graph_pooling

        #if self.num_layer < 2:
        #    raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.mean_pool = global_mean_pool
            self.max_pool = global_max_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
        self.linear1 = torch.nn.Linear(self.emb_dim*2, 32)
        self.linear1_1 = torch.nn.Linear(self.emb_dim, 32)
        self.linear2 = torch.nn.Linear(128, 32)
        self.linear3 = torch.nn.Linear(32, 16)
        self.graph_pred_linear = torch.nn.Linear(32, self.num_tasks)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, data, do_visualize=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        print('batch!!',batch)
        node_representation = self.gnn(x, edge_index, edge_attr)
        
        if self.graph_pooling == 'mean': 
            mean_pooled = self.mean_pool(node_representation, batch)
            max_pooled = self.max_pool(node_representation, batch)
            pooled = torch.cat((mean_pooled, max_pooled), dim=1)
        else:
            pooled = self.pool(node_representation, batch)
        
        #graph_rep = torch.cat([pooled, center_node_rep], dim = 1)
        if self.graph_pooling == 'mean': 
            graph_rep = self.linear1(pooled)
        else:
            graph_rep = self.linear1_1(pooled)
        #graph_rep = self.linear2(graph_rep)
        #graph_rep = self.linear3(graph_rep)
        if do_visualize:
            visualize(graph_rep, data.y)      
  
        return self.softmax(self.graph_pred_linear(graph_rep))


class GNN_graphpred_for_x(torch.nn.Module):
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

        #if self.num_layer < 2:
        #    raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(100, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
        self.linear1 = torch.nn.Linear(self.emb_dim*2, 128)
        self.linear2 = torch.nn.Linear(128, 32)
        self.linear3 = torch.nn.Linear(32, 16)
        self.graph_pred_linear1 = torch.nn.Linear(16, self.num_tasks)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, x, edge_index, do_visualize=False):
        #x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        node_representation = self.gnn(x, edge_index, None)
        
        #pooled = self.pool(node_representation)
        pooled = self.concat((global_mean_pool(node_representation), global_max_pool(node_representation)), dim=1)
        
        #graph_rep = torch.cat([pooled, center_node_rep], dim = 1)
        graph_rep = self.linear(pooled)
        
        #if do_visualize:
        #    visualize(graph_rep, data.y)      
  
        return self.sigmoid(self.graph_pred_linear(graph_rep))

if __name__ == "__main__":
    pass



