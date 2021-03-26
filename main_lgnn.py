# -*- encoding: utf-8 -*-
'''
@File    :   main_lgnn.py
@Time    :   2021/03/25 10:49:06
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import argparse
from utils.SBM import SBM

import torch

from utils.dataset import SBM_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #######################################################
    #           dataset setting                       #
    #######################################################
    parser.add_argument("--num_examples_train", type=int, default=6000)
    parser.add_argument("--num_examples_test", type=int, default=1000)
    parser.add_argument("--edge_density", type=float, default=0.2)
    parser.add_argument('--p_SBM', type=float, default=0.8)
    parser.add_argument('--q_SBM', type=float, default=0.2)
    parser.add_argument("--N_train", type=int, default=50, help="num of nodes")
    parser.add_argument("--N_test", type=int, default=50)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--save_path_root", type=str, default="./data/")

    parser.add_argument("--J", type=int, default=2)

    #######################################################
    #           pytorch setting                           #
    #######################################################
    parser.add_argument("--cuda_id", type=int, default=0, help="-1 for cpu")
    parser.add_argument("--torch_seed", type=int, default=42)
    args = parser.parse_args()

    if torch.cuda.is_available() and args.cuda_id>=0:
        device = torch.device("cuda:{}".format(args.cuda_id))  
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.torch_seed)

    dataset_train = SBM_dataset(args.p_SBM, 
                                args.q_SBM, 
                                graph_size=args.N_train,
                                n_classes=args.n_classes, 
                                num_graphs=args.num_examples_train,
                                J=args.J,
                                train=True, 
                                path_root=args.save_path_root, 
                                save_data=True)



