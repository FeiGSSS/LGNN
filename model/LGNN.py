# -*- encoding: utf-8 -*-
'''
@File    :   LGNN.py
@Time    :   2021/03/26 16:55:20
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class LGNN_layer(nn.Module):
    def __init__(self, size_Fa, 
                       size_Fb,
                       feat_dim_in,
                       feat_dim_line_in,
                       feat_dim_out,
                       feat_dim_line_out,
                       final_layer=False):
        super(LGNN_layer, self).__init__()
        self.size_Fa = size_Fa
        self.size_Fb = size_Fb

        self.final_layer = final_layer

        self.x2x_1 = nn.Linear(size_Fa*feat_dim_in, feat_dim_out, bias=False)
        self.y2x_1 = nn.Linear(2*feat_dim_line_in, feat_dim_out, bias=False)
        if not self.final_layer:
            self.x2x_2 = nn.Linear(size_Fa*feat_dim_in, feat_dim_out, bias=False)
            self.y2x_2 = nn.Linear(2*feat_dim_line_in, feat_dim_out, bias=False)

            self.y2y_1 = nn.Linear(size_Fb*feat_dim_line_in, feat_dim_line_out, bias=False)
            self.y2y_2 = nn.Linear(size_Fb*feat_dim_line_in, feat_dim_line_out, bias=False)
            self.x2y_1 = nn.Linear(4*feat_dim_out, feat_dim_line_out, bias=False)
            self.x2y_2 = nn.Linear(4*feat_dim_out, feat_dim_line_out, bias=False)

            self.bn_x = nn.BatchNorm1d(2*feat_dim_out)
            self.bn_y = nn.BatchNorm1d(2*feat_dim_line_out)



    def forward(self, inputs):
        """One forward pass step

        Args:
            Fa ([type]): list of original graphs' operations
            node_feat ([type]): nodes' feature
            Fb ([type]): list of line graphs' operators
            node_feat_line ([type]): ndoes feat in line graph
        """
        Fa, node_feat, Fb, node_feat_line, Pm, Pd = inputs
        assert self.size_Fa == len(Fa)
        assert self.size_Fb == len(Fb)
        if not self.final_layer:
            # Agg to X
            Fab = [Pm, Pd]
            x2x_by_operator = self.operatorXfeat(Fa, node_feat)
            y2x_by_operator = self.operatorXfeat(Fab, node_feat_line)
            z = F.relu(self.x2x_1(x2x_by_operator) + self.y2x_1(y2x_by_operator))
            z_prime = self.x2x_2(x2x_by_operator) + self.y2x_2(y2x_by_operator)
            x = torch.cat((z, z_prime), dim=1)
            x = self.bn_x(x)

            # Agg to y
            Fab_prime = [Pm.t(), Pd.t()]
            y2y_by_operator = self.operatorXfeat(Fb, node_feat_line)
            x2y_by_operator = self.operatorXfeat(Fab_prime, x)
            w = F.relu(self.y2y_1(y2y_by_operator) + self.x2y_1(x2y_by_operator))
            w_prime = self.y2y_2(y2y_by_operator) + self.x2y_2(x2y_by_operator)
            y = torch.cat((w, w_prime), dim=1)
            y = self.bn_y(y)

            return [Fa, x, Fb, y, Pm, Pd]
        else:
            Fab = [Pm, Pd]
            x2x_by_operator = self.operatorXfeat(Fa, node_feat)
            y2x_by_operator = self.operatorXfeat(Fab, node_feat_line)
            z_prime = self.x2x_1(x2x_by_operator) + self.y2x_1(y2x_by_operator)
            return z_prime

    def operatorXfeat(self, operators_list, feat):
        feat_out = []
        for A in operators_list:
            feat_out.append(torch.sparse.mm(A, feat))
        return torch.cat(feat_out, dim=1) # [N, (J+2)*feat_dim]
            

class LGNN(nn.Module):
    def __init__(self, hid_dim, num_layers, J, num_classes, device):
        super(LGNN, self).__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.J = J
        self.num_classes = num_classes
        self.device = device
        self.layers = self.build_layers()
        self.to(self.device)
        
    def forward(self, inputs):
        # Fa, node_feat, Fb, node_feat_line, Pm, Pd = inputs
        return self.layers(inputs)

    def build_layers(self):
        layers = nn.Sequential()
        layer = LGNN_layer(size_Fa=self.J+2,
                            size_Fb=self.J+2,
                            feat_dim_in=1,
                            feat_dim_out=int(self.hid_dim/2),
                            feat_dim_line_in=1,
                            feat_dim_line_out=int(self.hid_dim/2))
        layers.add_module(name="lgnn layer init", module=layer)
        
        for i in torch.arange(self.num_layers):
            layer = LGNN_layer(size_Fa=self.J+2,
                                size_Fb=self.J+2,
                                feat_dim_in=self.hid_dim,
                                feat_dim_out=int(self.hid_dim/2),
                                feat_dim_line_in=self.hid_dim,
                                feat_dim_line_out=int(self.hid_dim/2))
            layers.add_module(name="lgnn layer {}".format(i), module=layer)  
        
        layer = LGNN_layer(size_Fa=self.J+2,
                           size_Fb=self.J+2,
                           feat_dim_in=self.hid_dim,
                           feat_dim_out=self.num_classes,
                           feat_dim_line_in=self.hid_dim,
                           feat_dim_line_out=None,
                           final_layer=True)
        layers.add_module(name="lgnn layer final", module=layer)

        return layers  
        
