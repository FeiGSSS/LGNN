# LGNN: Line Graph Neural Networks
Pytorch re-implementation of Line Graph Neural Networks from ICLR 2019 paper [Supervised community detection with graph neural networks](https://arxiv.org/pdf/1705.08415.pdf). This implementation is based on the [offical codes](https://github.com/zhengdao-chen/GNN4CD). 

- The original implementation is pretty slow cause the data preprocessing in each iteration (about 6s). By utilizing `Dataset` and `DataLoader` in Pytorch, the running time of each iteration is reduced to 1.7s.
- Utilizing `torch.sparse` to perform matrix multiplication in GNN layers.

## Requirements
I recommend using an independent environment to run the codes:
```
conda create -n LGNN
conda activate LGNN
pip install -r requirements.txt
```
## Dataset
Before you run the code, you should make a new folder name `data` to store the generated SBM graph dataset. The dataset is named as `"sbm_{}_{}_{}_{}_{}_{}.pkl".format(p, q, N, n_classes, J, str(train))` .
## Train the Model
The key to accelerating the training process is utilizing the multi-processing of `DataLoader`, therefore, the `num_workers` should as large as possible. And the defalut arguments will train LGNN on 5-community dissociative SBM graphs with `n = 400, C = 5, p = 0, q = 18/n`.

Runing Example:
```
python main_lgnn.py --num_workers 40
```
