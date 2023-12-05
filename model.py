# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:06:28 2023

@author: Zhao Li
# @File: The HJE Model
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_hit1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def forward(self, pred1, tar1):
        pred1 = F.softmax(pred1, dim=1)
        loss = -torch.log(pred1[tar1 == 1]).sum()
        return loss

class HJE(BaseClass):

    def __init__(self, n_ent, n_rel, emb_dim, emb_dim1, max_arity, device):
        super(HJE, self).__init__()
        self.loss = MyLoss()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.device = device
        self.emb_dim = emb_dim
        self.emb_dim1 = emb_dim1
        self.emb_dim2 = emb_dim // emb_dim1
        self.lmbda = 0.05
        self.max_arity = max_arity
        self.ent_embeddings = nn.Parameter(torch.Tensor(self.n_ent, self.emb_dim))
        self.rel_embeddings = nn.Parameter(torch.Tensor(self.n_rel, self.emb_dim))
        self.pos_embeddings = nn.Embedding(self.max_arity, self.emb_dim)

        self.conv_layer_2 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 2))
        self.conv_layer_3 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 3))
        self.conv_layer_4 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 4))
        self.conv_layer_5 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 5))
        self.conv_layer_6 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 6))
        self.conv_layer_7 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 7))
        self.conv_layer_8 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 8))
        self.conv_layer_9 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 9))
        self.fc_rel_2 = nn.Linear(in_features=self.emb_dim, out_features=16)
        
        self.pool = torch.nn.MaxPool3d((4, 1, 1))
        self.pool2d = torch.nn.MaxPool2d((1, 2))

        self.inp_drop = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)
        self.dropout_3d = nn.Dropout(0.3)
        self.dropout_2d = nn.Dropout(0.3)
        self.nonlinear = nn.ReLU()
        self.conv_size = (self.emb_dim1 * self.emb_dim2) * 8 // 4
        self.conv_size_2d = (self.emb_dim) * 16 // 2
        self.fc_layer = nn.Linear(in_features=self.conv_size, out_features=self.emb_dim)
        self.W = torch.nn.Parameter(torch.empty(size=(2 * emb_dim, emb_dim)))
        self.fc_1 = nn.Linear(in_features=1*self.conv_size_2d, out_features=self.conv_size)
        self.fc_2 = nn.Linear(in_features=2*self.conv_size_2d, out_features=self.conv_size)
        self.fc_3 = nn.Linear(in_features=3*self.conv_size_2d, out_features=self.conv_size)
        self.fc_4 = nn.Linear(in_features=4*self.conv_size_2d, out_features=self.conv_size)
        self.fc_5 = nn.Linear(in_features=5*self.conv_size_2d, out_features=self.conv_size)
        self.fc_6 = nn.Linear(in_features=6*self.conv_size_2d, out_features=self.conv_size)
        self.fc_7 = nn.Linear(in_features=7*self.conv_size_2d, out_features=self.conv_size)
        self.fc_8 = nn.Linear(in_features=8*self.conv_size_2d, out_features=self.conv_size)

        self.bn1 = nn.BatchNorm3d(num_features=1)
        self.bn2 = nn.BatchNorm3d(num_features=4)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.bn4 = nn.BatchNorm1d(num_features=self.conv_size)
        self.register_parameter('b', nn.Parameter(torch.zeros(n_ent)))
        self.criterion = nn.Softplus()

        # 初始化 embeddings 以及卷积层、全连接层的参数
        nn.init.xavier_uniform_(self.ent_embeddings.data)
        nn.init.xavier_uniform_(self.rel_embeddings.data)
        nn.init.xavier_uniform_(self.pos_embeddings.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_2.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_3.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_4.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_5.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_6.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_7.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_8.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_9.weight.data)
        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        nn.init.xavier_uniform_(self.fc_rel_2.weight.data)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.fc_1.weight.data)
        nn.init.xavier_uniform_(self.fc_2.weight.data)
        nn.init.xavier_uniform_(self.fc_3.weight.data)
        nn.init.xavier_uniform_(self.fc_4.weight.data)
        nn.init.xavier_uniform_(self.fc_5.weight.data)
        nn.init.xavier_uniform_(self.fc_6.weight.data)
        nn.init.xavier_uniform_(self.fc_7.weight.data)
        nn.init.xavier_uniform_(self.fc_8.weight.data)



    def shift(self, v, sh):
        y = torch.cat((v[:, sh:], v[:, :sh]), dim=1)
        return y

    def conv3d_process(self, concat_input):
        r = concat_input[:, 0, :].view(-1, 1, self.emb_dim1, self.emb_dim2)

        if concat_input.shape[1] == 2:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_2(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 3:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_3(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 4:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_4(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 5:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_5(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 6:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_6(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 7:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_7(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 8:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e7 = concat_input[:, 7, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6, e7), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_8(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if concat_input.shape[1] == 9:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e7 = concat_input[:, 7, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e8 = concat_input[:, 8, :].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6, e7, e8), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_9(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        x = x.view(-1, self.conv_size)
        x = self.dropout_3d(x)
        return x

    def convolve(self, e_emb, r_emb):

        x = e_emb.view(e_emb.shape[0], 1, 1, -1)
        x = self.inp_drop(x)

        k1 = self.fc_rel_2(r_emb)
        k1 = k1.view(-1, 1, 16, 1, 1)
        k1 = k1.view(e_emb.size(0)*16, 1, 1, 1)
        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, k1, groups=e_emb.size(0))
        x = x.view(e_emb.size(0), 1, 1, 1, -1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

    def conv2d_process(self, concat_input):
        r = concat_input[:, 0, :]
        if concat_input.shape[1] == 2:
            e1 = concat_input[:, 1, :]
            x = self.convolve(e1, r)
            x = self.pool2d(x)
            x = x.contiguous().view(e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_1(x)
        if concat_input.shape[1] == 3:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            x = torch.cat((conv_e1, conv_e2), dim=1)
            x = self.pool2d(x)
            x = x.contiguous().view(e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_2(x)
        if concat_input.shape[1] == 4:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            x = torch.cat((conv_e1, conv_e2, conv_e3), dim=1)
            x = self.pool2d(x)
            x = x.contiguous().view(e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_3(x)
        if concat_input.shape[1] == 5:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            conv_e4 = self.convolve(e4, r)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4), dim=1)
            x = self.pool2d(x)
            x = x.contiguous().view(e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_4(x)
        if concat_input.shape[1] == 6:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            conv_e4 = self.convolve(e4, r)
            conv_e5 = self.convolve(e5, r)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5), dim=1)
            x = self.pool2d(x)
            x = x.contiguous().view(e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_5(x)
        if concat_input.shape[1] == 7:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            e6 = concat_input[:, 6, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            conv_e4 = self.convolve(e4, r)
            conv_e5 = self.convolve(e5, r)
            conv_e6 = self.convolve(e6, r)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6), dim=1)
            x = self.pool2d(x)
            x = x.contiguous().view(e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_6(x)
        if concat_input.shape[1] == 8:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            e6 = concat_input[:, 6, :]
            e7 = concat_input[:, 7, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            conv_e4 = self.convolve(e4, r)
            conv_e5 = self.convolve(e5, r)
            conv_e6 = self.convolve(e6, r)
            conv_e7 = self.convolve(e7, r)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7), dim=1)
            x = self.pool2d(x)
            x = x.contiguous().view(e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_7(x)
        if concat_input.shape[1] == 9:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            e6 = concat_input[:, 6, :]
            e7 = concat_input[:, 7, :]
            e8 = concat_input[:, 8, :]
            conv_e1 = self.convolve(e1, r)
            conv_e2 = self.convolve(e2, r)
            conv_e3 = self.convolve(e3, r)
            conv_e4 = self.convolve(e4, r)
            conv_e5 = self.convolve(e5, r)
            conv_e6 = self.convolve(e6, r)
            conv_e7 = self.convolve(e7, r)
            conv_e8 = self.convolve(e8, r)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7, conv_e8), dim=1)
            x = self.pool2d(x)
            x = x.contiguous().view(e1.shape[0], -1)
            x = self.dropout(x)
            x = self.fc_8(x)


        return x

    def forward(self, rel_idx, ent_idx, miss_ent_domain):

        r = self.rel_embeddings[rel_idx].unsqueeze(1)
        ents = self.ent_embeddings[ent_idx]
        pos = [i for i in range(ent_idx.shape[1]+1) if i + 1 != miss_ent_domain]
        pos = torch.tensor(pos).to(self.device)
        pos = pos.unsqueeze(0).expand_as(ent_idx)
        ents = ents + self.pos_embeddings(pos)
        concat_input = torch.cat((r, ents), dim=1)

        v1 = self.conv3d_process(concat_input)
        v2 = self.conv2d_process(concat_input)
        x = v1 + v2
        # x = v2
        # x = self.bn4(x)
        # x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.fc_layer(x)

        miss_ent_domain = torch.LongTensor([miss_ent_domain-1]).to(self.device)
        mis_pos = self.pos_embeddings(miss_ent_domain)
        tar_emb = self.ent_embeddings + mis_pos
        scores = torch.mm(x, tar_emb.transpose(0, 1))
        scores += self.b.expand_as(scores)

        return scores




    # def predict(self, test_batch):

