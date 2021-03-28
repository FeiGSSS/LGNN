
import os
import copy
import tqdm
import pickle as pkl
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset

from utils.SBM import SBM

class SBM_dataset(Dataset):
    def __init__(self, 
                p_SBM, 
                q_SBM, 
                graph_size,
                n_classes,
                num_graphs,
                J,
                train=True,
                path_root="./data/",
                save_data=True):
        self.p = p_SBM
        self.q = q_SBM
        self.N = graph_size
        self.n_classes = n_classes
        self.n_graphs = num_graphs
        self.J = J
        self.train = train
        self.path_root = path_root
        self.save_data = save_data

        self.save_path = self._save_path()
        self._generate_graphs()

    def _save_path(self):
        data_conf = "sbm_{}_{}_{}_{}_{}_{}.pkl".format(self.p, 
                                            self.q, 
                                            self.N, 
                                            self.n_classes,
                                            self.J,
                                            str(self.train))
        save_path = os.path.join(self.path_root, data_conf)
        return save_path

    def _generate_graphs(self):
        if os.path.exists(self.save_path):
            print("Loading sbm dataset from {}".format(self.save_path))
            with open(self.save_path, "rb") as f:
                self.dataset = pkl.load(f)
        else:
            print("Generate smb datasets ...")
            sbm_graphs = []
            for i in range(self.n_graphs):
                adj, label = SBM(self.p, self.q, self.N, self.n_classes)
                sbm_graphs.append([adj, label])
            self.dataset = sbm_graphs

            if self.save_data:
                print("Saving dataset to {}".format(self.save_path))
                with open(self.save_path, "wb") as f:
                    pkl.dump(self.dataset, f)

    @staticmethod
    def compute_operators(adj, J):
        """Generate I, D, A, A2, et, al.
        """
        num_nodes = adj.shape[0]
        degrees = np.sum(adj, axis=1)
        degree_matrix = np.diag(degrees)

        A = copy.deepcopy(adj)
        operators = [None] * (J+2)
        operators[0] = np.eye(num_nodes)
        for j in range(J):
            operators[j+1] = copy.deepcopy(A)
            A = np.minimum(np.matmul(A, A), np.ones_like(A))
        operators[J+1] = degree_matrix
        x = degrees.reshape(num_nodes, 1)
        return operators, x

    @staticmethod
    def Pm_Pd(adj):
        """Generate a [N, 2*E] matrix Pm and Pd, 
        Pm[n_i][e_k] = 1 if node n_i is one of end nodes of directed edge e_k
        otherwise, Pm[n_i][e_k] = 0

        Pd[n_i][e_k] = 1 if node n_i is source node of directed edge e_k
        Pd[n_i][e_k] = -1 if node n_i is target node of directed edge e_k
        otherwise, Pd[n_i][e_k] = 0

        Args:
            adj ([type]): syc, binary adj
        """
        num_nodes = adj.shape[0]
        num_edges = int(adj.sum() // 2)
        assert num_edges*2 == adj.sum()

        Pm = np.zeros((num_nodes, num_edges*2))
        Pd = np.zeros((num_nodes, num_edges*2))

        edge_count = 0
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adj[i][j] == 1:
                    Pm[i][edge_count] = 1
                    Pm[j][edge_count] = 1
                    Pm[i][edge_count+num_edges] = 1
                    Pm[j][edge_count+num_edges] = 1

                    Pd[i][edge_count] = 1
                    Pd[j][edge_count] = -1
                    Pd[i][edge_count+num_edges] = 1
                    Pd[j][edge_count+num_edges] = -1

                    edge_count += 1

        assert edge_count == num_edges

        return Pm, Pd

    def line_graph_adj(self, Pm, Pd):
        Pf = (Pm + Pd) / 2 # Pf[n_i][e_k] = 1 if node i is source node of directed edge k, otherwise 0
        Pt = (Pm - Pd) / 2 # Pf[n_i][e_k] = 1 if node i is target node of directed edge k, otherwise 0
        NB_matrix = np.matmul(Pt.transpose(), Pf) * (1 - np.matmul(Pf.transpose(), Pt))
        return NB_matrix

    def to_sparse_data(self, dense_adj_list):
        if not isinstance(dense_adj_list, list):
            dense_adj_list = [dense_adj_list]
        sparse_adj_list = []
        for adj in dense_adj_list:
            data = sp.coo_matrix(adj)
            sparse_adj_list.append(data)
        return sparse_adj_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        adj, label  = self.dataset[idx]
        #sbm_graphs.append((adj, label))
        Pm, Pd = self.Pm_Pd(adj)
        adj_line = self.line_graph_adj(Pm ,Pd)
        operators, node_feat = self.compute_operators(adj, self.J)
        operators_line, node_feat_line = self.compute_operators(adj_line, self.J)

        # dense matrix to pyg
        operators = self.to_sparse_data(operators)
        operators_line = self.to_sparse_data(operators_line)
        Pm = self.to_sparse_data(Pm)[0]
        Pd = self.to_sparse_data(Pd)[0]

        Graph = {"label": label,
                "operators": operators,
                "node_feat": node_feat, 
                "operators_line": operators_line,
                "node_feat_line": node_feat_line,
                "Pm": Pm,
                "Pd": Pd}
        return Graph

    @staticmethod
    def collate_fn(sample_list):
        assert isinstance(sample_list, list)
        assert len(sample_list) == 1 ; "Only support batch_size=1"
        return sample_list[0]
        


