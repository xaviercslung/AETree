from torch.utils.data import Dataset
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import numpy as np

import math
import numpy as np
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import sys

sys.path.append('../')

import copy
import time
from tensorboardX import SummaryWriter

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel('ERROR')


def rotate_xy_2(p, sin, cos, center):
    x_ = (p[:, 0:1] - center[:, 0:1]) * cos - (p[:, 1:2] - center[:, 1:2]) * sin + center[:, 0:1]
    y_ = (p[:, 0:1] - center[:, 0:1]) * sin + (p[:, 1:2] - center[:, 1:2]) * cos + center[:, 1:2]
    #     print(((p[:,0:1]-center[:,0:1])*cos).shape, cos.shape, x_.shape)
    return np.hstack((x_, y_))


def get_box_2(P, F):
    ld = np.hstack((P[:, 0:1] - F[:, 0:1] / 2, P[:, 1:2] - F[:, 1:2] / 2))
    rd = np.hstack((P[:, 0:1] + F[:, 0:1] / 2, P[:, 1:2] - F[:, 1:2] / 2))
    ru = np.hstack((P[:, 0:1] + F[:, 0:1] / 2, P[:, 1:2] + F[:, 1:2] / 2))
    lu = np.hstack((P[:, 0:1] - F[:, 0:1] / 2, P[:, 1:2] + F[:, 1:2] / 2))
    # box = np.hstack((ld, rd, ru, lu)).reshape(len(P), -1, 2)
    sinO = np.sin(F[:, 2:3])
    cosO = np.cos(F[:, 2:3])

    ld_r = rotate_xy_2(ld, sinO, cosO, P)
    rd_r = rotate_xy_2(rd, sinO, cosO, P)
    ru_r = rotate_xy_2(ru, sinO, cosO, P)
    lu_r = rotate_xy_2(lu, sinO, cosO, P)
    if (len(P) > 0):
        box_r = np.hstack((ld_r, rd_r, ru_r, lu_r)).reshape(len(P), -1, 2)
    else:
        box_r = []
    return box_r


#     plt.figure(figsize=(15, 15))
#     for i, p in enumerate(box_r):
#         draw_polygon_c(p, i, P[i,:2],'r')
# #     plt.savefig('ab.jpg')
#     plt.show()

def draw_polygon_c(pc, txt, center, color):
    X, Y = pc[:, 0], pc[:, 1]
    plt.plot(X, Y, c=color)
    plt.plot([X[-1], X[0]], [Y[-1], Y[0]], c=color)
    if (txt):
        plt.annotate(txt, center, size=8)


#     plt.scatter(center[0], center[1], c='b')
#     n = np.arange(len(pc))
#     for i,txt in enumerate(n):
#         plt.annotate(txt,(pc[i,0],pc[i,1]),size=8)


#     plt.axis('off')
#     plt.show()

def draw_box(box, txt, center, color):
    #     print(box.shape)
    for i, p in enumerate(box):
        #         print(p.shape)
        c = 'b' if color[i] else 'r'
        draw_polygon_c(p, txt[i], center[i], c)


def plot_boxes(samples, label, center, color, n, m, test_num, save=False, savename='pclouds'):
    fig = plt.figure(figsize=(5 * m, 5 * n))
    fig.set_tight_layout(True)
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            ax = fig.add_subplot(n, m, idx + 1)
            #             print(idx)
            draw_box(samples[idx], label[idx], center[idx], color[idx])
    if save:
        plt.savefig('./log_' + test_num + '/' + savename)

    plt.show()


def inference(P, F, n, out=[]):
    '''
    Input:
        P: Positions of initial node  n*3
        F: Features of initial node  n*d
        n: Max level of generated tree
    Output:
        out: Position List of generated children nodes
    '''
    if (n == 0):
        return out
    P_list = []
    P_list_re = []
    I_list = []
    leaf_node_list = []

    tmp_P = P
    tmp_F = F
    father_I = torch.zeros((1, 1))
    P_list.append(P[0])
    print(n)
    for i in range(n):
        print('--------------', i, '---------------')
        left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf = model.decoder(tmp_F, tmp_P)
        print(left_featrue.shape, left_P.shape, left_isleaf.shape, right_featrue.shape, right_P.shape,
              right_isleaf.shape)
        print(left_P, left_isleaf, right_P, right_isleaf)

        left_xy_new = left_P[:, :2] * tmp_P[:, 2:4] + tmp_P[:, :2]
        left_P[:, :2] = left_xy_new
        left_wh_new = left_P[:, 2:4] * tmp_P[:, 2:4]
        left_P[:, 2:4] = left_wh_new
        left_a_new = left_P[:, 4] + tmp_P[:, 4]
        left_P[:, 4] = left_a_new

        right_xy_new = right_P[:, :2] * tmp_P[:, 2:4] + tmp_P[:, :2]
        right_P[:, :2] = right_xy_new
        right_wh_new = right_P[:, 2:4] * tmp_P[:, 2:4]
        right_P[:, 2:4] = right_wh_new
        right_a_new = right_P[:, 4] + tmp_P[:, 4]
        right_P[:, 4] = right_a_new

        I = np.zeros(len(left_P) * 2, dtype='int32')

        temp_I_list = []
        left_node_index = torch.zeros((len(left_P), 1))
        right_node_index = torch.zeros((len(right_P), 1))

        for j in range(len(left_P)):
            P_list.append(left_P[j])
            left_index = len(P_list) - 1
            P_list.append(right_P[j])
            right_index = len(P_list) - 1

            father_index = father_I[j].detach().numpy()
            father_index = int(father_index)
            temp_I = [left_index, right_index, father_index]

            I[2 * j] = left_index
            I[2 * j + 1] = right_index
            left_node_index[j] = left_index
            right_node_index[j] = right_index

            print('*******I:', temp_I)
            temp_I_list.append(temp_I)

        if (temp_I_list):
            I_list.append(temp_I_list)

        print(I)

        tmp_F = []
        tmp_P = []
        father_I = []

        print(left_isleaf, right_isleaf)

        left_isleaf = torch.round(left_isleaf)[:, 0]
        right_isleaf = torch.round(right_isleaf)[:, 0]

        print(left_isleaf, right_isleaf)

        tmp_F.append(left_featrue[left_isleaf == 0, :])
        tmp_P.append(left_P[left_isleaf == 0, :])
        father_I.append(left_node_index[left_isleaf == 0, :])

        tmp_F.append(right_featrue[right_isleaf == 0, :])
        tmp_P.append(right_P[right_isleaf == 0, :])
        father_I.append(right_node_index[right_isleaf == 0, :])

        print('lllllll', father_I)
        #         print(father_I[0].shape)
        tmp_F = torch.cat(tmp_F, 0)
        tmp_P = torch.cat(tmp_P, 0)
        father_I = torch.cat(father_I, 0)
        #         print(father_I[0].shape)

        print(tmp_F.shape)
        print(tmp_P.shape)

    P_list = torch.stack(P_list)
    return P_list, I_list


def get_last_box(root_X, root_F, n_layer):
    P_list, I_list = inference(root_X, root_F, n_layer)
    X_ab_r = P_list.detach().numpy()
    box_infer_list = []
    center_infer_list = []
    txt_infer_list = []
    color_infer_list = []

    n = 1
    node_index = np.arange(1)
    color = np.zeros(len(node_index))
    P = X_ab_r[node_index, :2]
    F = X_ab_r[node_index, 2:]
    box = get_box_2(P, F)
    box_infer_list.append(box)
    center_infer_list.append(P)
    txt_infer_list.append(node_index)
    color_infer_list.append(color)

    for Ii in I_list:
        Ii = np.array(Ii)
        left_idx = (Ii[:, 0]).astype('int32')
        right_idx = (Ii[:, 1]).astype('int32')
        father_idx = (Ii[:, 2]).astype('int32')
        #     print(left_idx, right_idx, father_idx,node_index)
        node_index = list(set(node_index) - set(father_idx) | set(left_idx) | set(right_idx))
        #     print(node_index)
        color = np.zeros(len(node_index))
        for idx in father_idx:
            color[(np.arange(len(node_index))[node_index == idx])] = 1

        if (len(node_index) > 0):
            P = X_ab_r[node_index, :2]
            F = X_ab_r[node_index, 2:]
            #         print(P,F)
            box = get_box_2(P, F)
            #         print(box)
            box_infer_list.append(box)
            center_infer_list.append(P)
            txt_infer_list.append(node_index)
            color_infer_list.append(color)
            n = n + 1
    return box_infer_list[-1], txt_infer_list[-1], center_infer_list[-1], color_infer_list[-1]
