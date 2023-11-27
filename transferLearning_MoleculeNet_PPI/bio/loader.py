import os
import torch
import random
import networkx as nx
import pandas as pd
import numpy as np
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
from collections import Counter, deque
from networkx.algorithms.traversal.breadth_first_search import generic_bfs_edges

import torch_geometric.utils as tg_utils
import networkx as nx

class BioDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data_type,
                 empty=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: the data directory that contains a raw and processed dir
        :param data_type: either supervised or unsupervised
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        :param transform:
        :param pre_transform:
        :param pre_filter:
        """
        self.root = root
        self.data_type = data_type

        super(BioDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if not empty:
            #아 데이터 전체랑, 전체 데이터를 각 개별 그래프로 끊는 slice
            self.data, self.slices = torch.load(self.processed_paths[0])
            print("what is processed_path: ", self.processed_paths)

    @property
    def raw_file_names(self):
        #raise NotImplementedError('Data is assumed to be processed')
        if self.data_type == 'supervised': # 8 labelled species
            file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090', '4932', '7955']
        else: # unsupervised: 8 labelled species, and 42 top unlabelled species by n_nodes.
            file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090',
            '4932', '7955', '3694', '39947', '10116', '443255', '9913', '13616',
            '3847', '4577', '8364', '9823', '9615', '9544', '9796', '3055', '7159',
            '9031', '7739', '395019', '88036', '9685', '9258', '9598', '485913',
            '44689', '9593', '7897', '31033', '749414', '59729', '536227', '4081',
            '8090', '9601', '749927', '13735', '448385', '457427', '3711', '479433',
            '479432', '28377', '9646']
        return file_name_list


    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        raise ('Data is assumed to be processed')

class BioDataset_graphcl(BioDataset):
    def __init__(self,
                 root,
                 data_type,
                 augmentation1=0,
                 augmentation2=0,
                 augmentation3=0,
                 empty=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        self.set_augMode('none')
        self.set_augStrength(0.2)
        self.set_augProb(None)
        self.augmentations = [self.node_drop, self.subgraph, self.edge_pert, self.attr_mask, lambda x:x]
        self.augmentations_v2 = [self.augmentations[augmentation1], self.augmentations[augmentation2], self.augmentations[augmentation3]]
        self.augmentations_v3 = [self.augmentations[augmentation1], self.augmentations[augmentation2]]
        super(BioDataset_graphcl, self).__init__(root, data_type, empty, transform, pre_transform, pre_filter)

    def set_aug(self, aug1, aug2):
        self.aug1 = int(aug1)
        self.aug2 = int(aug2)
        
    def set_augMode(self, aug_mode):
        self.aug_mode = aug_mode

    def set_augStrength(self, aug_strength):
        self.aug_strength = aug_strength

    def set_augProb(self, aug_prob):
        self.aug_prob = aug_prob

    def node_drop(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num  * self.aug_strength)

        idx_perm = np.random.permutation(node_num)
        idx_nondrop = idx_perm[drop_num:].tolist()
        idx_nondrop.sort()

        edge_index, edge_attr = tg_utils.subgraph(idx_nondrop, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes=node_num)

        data.x = data.x[idx_nondrop]
        data.edge_index = edge_index
        #data.edge_attr = edge_attr
        data.__num_nodes__, _ = data.x.shape
        return data

    def edge_pert(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        pert_num = int(edge_num * self.aug_strength)

        # delete edges
        idx_drop = np.random.choice(edge_num, (edge_num - pert_num), replace=False)
        edge_index = data.edge_index[:, idx_drop]
        #edge_attr = data.edge_attr[idx_drop]

        # add edges
        adj = torch.ones((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 0
        edge_index_nonexist = adj.nonzero(as_tuple=False).t()
        idx_add = np.random.choice(edge_index_nonexist.shape[1], pert_num, replace=False)
        edge_index_add = edge_index_nonexist[:, idx_add]
        # random 9-dim edge_attr, for details please refer to https://github.com/snap-stanford/pretrain-gnns/issues/30
        #edge_attr_add = torch.tensor( np.random.randint(2, size=(edge_index_add.shape[1], 7)), dtype=torch.float32 )
        #edge_attr_add = torch.cat((edge_attr_add, torch.zeros((edge_attr_add.shape[0], 2))), dim=1)
        edge_index = torch.cat((edge_index, edge_index_add), dim=1)
        #edge_attr = torch.cat((edge_attr, edge_attr_add), dim=0)

        data.edge_index = edge_index
        #data.edge_attr = edge_attr
        return data

    def attr_mask(self, data):
        node_num, _ = data.x.size()
        mask_num = int(node_num * self.aug_strength)
        _x = data.x.clone()

        token = data.x.mean(dim=0)
        idx_mask = np.random.choice(node_num, mask_num, replace=False)

        _x[idx_mask] = token
        data.x = _x
        return data

    def subgraph(self, data):
        G = tg_utils.to_networkx(data)

        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * (1-self.aug_strength))

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
                idx_neigh = set([np.random.choice(idx_unsub)])
            sample_node = np.random.choice(list(idx_neigh))

            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

        idx_nondrop = idx_sub
        idx_nondrop.sort()

        edge_index, edge_attr = tg_utils.subgraph(idx_nondrop, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes=node_num)

        data.x = data.x[idx_nondrop]
        data.edge_index = edge_index
        #data.edge_attr = edge_attr
        data.__num_nodes__, _ = data.x.shape
        return data

    def get(self, idx):
        data, data1, data2 = Data(), Data(), Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key], data1[key], data2[key] = item[s], item[s], item[s]
        #print("what is augmode: " , self.aug_mode)
        if self.aug_mode == 'none':
            n_aug1, n_aug2 = self.aug1, self.aug2
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'uniform':
            n_aug = np.random.choice(25, 1)[0]
            n_aug1, n_aug2 = n_aug//5, n_aug%5
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'sample':
            n_aug = np.random.choice(25, 1, p=self.aug_prob)[0]
            n_aug1, n_aug2 = n_aug//5, n_aug%5
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2) 
            #print(self.augmentations[n_aug1],self.augmentations[n_aug2])
        elif self.aug_mode == 'sample_v2':
            n_aug = np.random.choice(9, 1, p=self.aug_prob)[0]
            n_aug1, n_aug2 = n_aug//3, n_aug%3
            data1 = self.augmentations_v2[n_aug1](data1)
            data2 = self.augmentations_v2[n_aug2](data2) 
        elif self.aug_mode == 'sample_v3':
            n_aug = np.random.choice(4, 1, p=self.aug_prob)[0]
            n_aug1, n_aug2 = n_aug//2, n_aug%2
            data1 = self.augmentations_v3[n_aug1](data1)
            data2 = self.augmentations_v3[n_aug2](data2) 

        return data, data1, data2


def custom_collate(data_list):
    batch = Batch.from_data_list([d[0] for d in data_list], follow_batch=['edge_index', 'edge_index_neg'])
    batch_1 = Batch.from_data_list([d[1] for d in data_list])
    batch_2 = Batch.from_data_list([d[2] for d in data_list])
    return batch, batch_1, batch_2


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True,
                 **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=custom_collate, **kwargs)


if __name__ == "__main__":
    root_supervised = 'dataset/supervised'
    d_supervised = BioDataset(root_supervised, data_type='supervised')
    print(d_supervised)

    root_unsupervised = 'dataset/unsupervised'
    d_unsupervised = BioDataset(root_unsupervised, data_type='unsupervised')
    print(d_unsupervised)

