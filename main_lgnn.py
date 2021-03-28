# -*- encoding: utf-8 -*-
'''
@File    :   main_lgnn.py
@Time    :   2021/03/25 10:49:06
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.dataset import SBM_dataset
from model.LGNN import LGNN
from utils.losses import compute_loss_multiclass, compute_accuracy_multiclass 
from utils.utils import data_convert

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset setting
    parser.add_argument("--num_examples_train", type=int, default=6000)
    parser.add_argument("--num_examples_test", type=int, default=100)
    parser.add_argument("--edge_density", type=float, default=0.2)
    parser.add_argument('--p_SBM', type=float, default=0.0)
    parser.add_argument('--q_SBM', type=float, default=0.045)
    parser.add_argument("--N_train", type=int, default=400, help="num of nodes")
    parser.add_argument("--N_test", type=int, default=400)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--save_path_root", type=str, default="./data/")
    # GNN parameters
    parser.add_argument("--J", type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=30)
    parser.add_argument("--lr", type=float, default=4e-3)
    # pytorch setting                           #
    parser.add_argument("--cuda_id", type=int, default=0, help="-1 for cpu")
    parser.add_argument("--torch_seed", type=int, default=42)
    parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float, default=40.0)
    # experiments setting
    parser.add_argument("--show_freq", type=int, default=10)
    parser.add_argument("--model_save_path", type=str, default="./model_ckp/")
    args = parser.parse_args()

    if torch.cuda.is_available() and args.cuda_id>=0:
        device = torch.device("cuda:{}".format(args.cuda_id))  
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.torch_seed)

    # Dataset
    dataset_train = SBM_dataset(args.p_SBM, 
                                args.q_SBM, 
                                graph_size=args.N_train,
                                n_classes=args.n_classes, 
                                num_graphs=args.num_examples_train,
                                J=args.J,
                                train=True, 
                                path_root=args.save_path_root, 
                                save_data=True)
    dataloader_train = DataLoader(dataset_train, 
                                  batch_size=1, 
                                  num_workers=18,
                                  shuffle=True, 
                                  collate_fn=dataset_train.collate_fn)
    # Model
    model = LGNN(hid_dim=args.hid_dim,
                num_layers=args.num_layers,
                J=args.J,
                num_classes=args.n_classes,
                device=device)

    # Optimizer
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    losses = []
    acces = []
    t0 = time.time()
    for cnt, data in enumerate(dataloader_train):
        optimizer.zero_grad()
        inputs = data_convert(data, device)
        pred = model(inputs) # [N, n_classes]

        pred = pred[None, :, :]
        label = torch.Tensor(data["label"])[None, :]

        loss = compute_loss_multiclass(pred, label, args.n_classes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        acc = compute_accuracy_multiclass(pred, label, args.n_classes)

        losses.append(loss.item())
        acces.append(acc.item())

        t1 = time.time()
        if (cnt+1)%args.show_freq == 0 :
            ave_loss = np.mean(losses[-args.show_freq:])
            ave_acc = np.mean(acces[-args.show_freq:])
            print("Sample : {:<3}, Loss={:.3f}, ACC={:.4f}, cmu_Time={:.1f}s".format(cnt, ave_loss, ave_acc, t1-t0))

        




