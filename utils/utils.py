import torch
import numpy as np

def data_convert(data_dict, device):
    Fa = [scipy2pytorch(x).to(device) for x in data_dict["operators"]]
    node_feat = torch.FloatTensor(data_dict["node_feat"]).to(device)
    Fb = [scipy2pytorch(x).to(device) for x in data_dict["operators_line"]]
    node_feat_line = torch.FloatTensor(data_dict["node_feat_line"]).to(device)
    Pm = scipy2pytorch(data_dict["Pm"]).to(device)
    Pd = scipy2pytorch(data_dict["Pd"]).to(device)
    return Fa, node_feat, Fb, node_feat_line, Pm, Pd

def scipy2pytorch(sp):
    values = sp.data
    indices = np.vstack((sp.row, sp.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sp.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))