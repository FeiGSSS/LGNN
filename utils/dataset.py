import os
import pickle as pkl

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
        self.train = train
        self.path_root = path_root
        self.save_data = save_data

        self.save_path = self._save_path()
        self._generate_graphs()

    def _save_path(self):
        data_conf = "sbm_{}_{}_{}_{}_{}.pkl".format(self.p, 
                                            self.q, 
                                            self.N, 
                                            self.n_classes,
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
                sbm_graphs.append((adj, label))
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
        degrees = torch.sum(adj, dim=1)
        degree_matrix = torch.diag(degrees)

        A = adj.copy()
        operators = torch.zeros((num_nodes, num_nodes, J+2))
        operators[:, :, 0] = torch.eye(num_nodes)
        for j in range(J):
            operators[:, :, j+1] = A.copy()
            A = torch.minimum(torch.mm(A, A), torch.ones_like(A))
        operators[:, :, J+1] = degree_matrix
        x = degrees.reshape(num_nodes, 1)
        return operators, x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        adj, label = self.dataset[idx]
        return adj, label
                


