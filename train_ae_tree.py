#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import sys
sys.path.append('./model/')
from model_ae_tree import * # which file are we importing?
import copy
import time
from tensorboardX import SummaryWriter

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

def draw_point(ax, pc, min_xy, txt, s, color='b'):
#     plt.figure(num=3, figsize=(5, 5))
#     fig = plt.figure()
    
    for i in range(len(pc)):
        X, Y= pc[i,:2]
        mx, my = min_xy[i,:2]
        scale = pc[i,2]
        plt.scatter(X, Y, c=color,s=s+20)
        plt.annotate(txt[i], pc[i,:2], size=s)  
        rect = patches.Rectangle((mx, my), scale, scale,linewidth=scale*4, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
    plt.axis('equal')

def plot_tree(X, I, min_xy):
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1,1,1)
    colors = [plt.cm.hsv(i/float(len(I)+1)) for i in range(len(I)+1)]
    leaf_idx = np.arange(64)
    draw_point(ax, X[leaf_idx], min_xy[leaf_idx], leaf_idx, 10, colors[0])

    for i, Ii in enumerate(I):
#         print(i, Ii)
        Ii = Ii.squeeze(0).detach().numpy()
        node_index = (Ii[Ii[:,2] < 127][:,2]).astype('int32')
        draw_point(ax, X[node_index], min_xy[node_index], node_index, 10+2*i, colors[i+1])
#     plt.show()

def draw_point2(pc, label):
    for i in range(len(pc)):
        X, Y= pc[i]
        txt = label[i]
        plt.scatter(X, Y, c='b')
        plt.annotate(str(txt), pc[i], size=12)
    plt.axis('equal')
    my_x_ticks = np.arange(0, 1.2, 0.2)
    my_y_ticks = np.arange(0, 1.2, 0.2)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
#     plt.axis('on')
    
def plot_samples(samples, label, n, m, save=False, savename='pclouds'):
    fig = plt.figure(figsize=(5*m,5*n))
    fig.set_tight_layout(True)
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            ax = fig.add_subplot(n, m, idx+1)
#             print(idx)
            draw_point2(samples[idx], label[idx])  
    if save:
        plt.savefig(savename)

    plt.show()

def get_minxy_s_list(X_tree0, X_tree0_r, I_list):
    
    child_list = [[] for _ in range(len(X_tree0))]
    s_list = np.zeros((X_tree0.shape[0], 1))
    min_xy_list = np.zeros((X_tree0.shape[0], 2))
    min_xy_list[:64] = X_tree0[:64,:2]

    s_list2 = np.zeros((X_tree0_r.shape[0], 1))
    min_xy_list2 = np.zeros((X_tree0_r.shape[0], 2))
    min_xy_list2[:64] = X_tree0_r[:64,:2]

    for i, Ii in enumerate(I_list):
        Ii = Ii.squeeze(0).detach().numpy()

        node_index = (Ii[Ii[:,2] < 127]).astype('int32')
    #     print(node_index)|
        child_node = node_index[:,:2]
        father_node =  node_index[:,2]
        
        for i,idx in enumerate(father_node):
            tmp_child = []
            for node_id in child_node[i]:
                if(len(child_list[int(node_id)])>0):
                    for child in child_list[int(node_id)]:
                        tmp_child.append(child)
                else: 
                    tmp_child.append(int(node_id))
            child_xy = X_tree0[tmp_child,:2]
            min_xy_list[int(idx)] = np.min(child_xy, axis=0)
            s_list[int(idx)] = max(np.max(child_xy, axis=0) - np.min(child_xy, axis=0))
            
            child_xy2 = X_tree0_r[tmp_child,:2]
            min_xy_list2[int(idx)] = np.min(child_xy2, axis=0)
            s_list2[int(idx)] = max(np.max(child_xy2, axis=0) - np.min(child_xy2, axis=0))

            child_list[int(idx)] = tmp_child

    return min_xy_list, s_list, min_xy_list2, s_list2

def train_unsupervised(model, optimizer, scheduler, train_loader, test_loader, device, loss_save_dir,
                       num_epochs=100, M=1):
    print('Training your model!\n')
    model.train()

    best_params = None
    best_loss = float('inf')
    logs = defaultdict(list)
    
    try:
        for epoch in range(num_epochs):

            for i,(node_xys, I_list, node_fea, node_is_leaf) in enumerate(train_loader, 0):
                optimizer.zero_grad()
#                 print(i)
                node_xys = node_xys.to(device)
                node_xys = node_xys.float()
                I_list = [t.to(device) for t in I_list ]
                node_fea = node_fea.to(device)
                node_is_leaf = node_is_leaf.to(device)
                
                loss, _, _, _, _, _ = model(node_xys,node_fea,I_list,node_is_leaf)
                loss.backward()
                
#                 for name, parms in model.named_parameters():
#                     print('-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad,' -->leaf:', parms.is_leaf)
                
#                 torch.nn.utils.clip_grad_norm(model.parameters(), 5)
                lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = max(lr, 1e-5)
                
                optimizer.step()
            scheduler.step()
            
            if (epoch % 1) == 0 or epoch == num_epochs-1:
                start_time = time.time()
                train_loss, train_loss_rec, train_loss_rec_xy, train_loss_rec_s, train_loss_p, train_loss_leaf = model.loss_on_loader(train_loader, device)
                end_time = time.time()
                train_time = end_time - start_time
                
                start_time = time.time()
                test_loss, test_loss_rec, test_loss_rec_xy, test_loss_rec_s, test_loss_p, test_loss_leaf = model.loss_on_loader(test_loader, device)
                end_time = time.time()
                test_time = end_time - start_time
                
                logs['train_loss'].append(train_loss)
                logs['test_loss'].append(test_loss)
                
                writer.add_scalars('ae',
                                   {'train_loss': train_loss,
                                    'train_loss_rec': train_loss_rec,
                                    'train_loss_rec_xy': train_loss_rec_xy,
                                    'train_loss_rec_s': train_loss_rec_s,
                                    'train_loss_p': train_loss_p,
                                    'train_loss_leaf': train_loss_leaf,
                                    'test_loss': test_loss,
                                   'test_loss_rec': test_loss_rec,
                                    'test_loss_rec_xy': test_loss_rec_xy,
                                    'test_loss_rec_s': test_loss_rec_s,
                                   'test_loss_p': test_loss_p,
                                   'test_loss_leaf': test_loss_leaf,}, epoch)
                
                if (torch.isnan(test_loss)):
                    print("Epoch {epoch} Loss in nan!!!".format(epoch=epoch))
                    break;
                else:
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_params = copy.deepcopy(model.state_dict())

                    print("Epoch {epoch} Lr={Lr}, train loss={train_loss}, traintime={train_time}, test loss={test_loss}, test time={test_time}"
                           .format(epoch=epoch, Lr= lr, train_loss=train_loss, train_time=train_time, test_loss=test_loss, test_time=test_time)
                         )

    except KeyboardInterrupt:
        pass
#     print(best_loss, best_params)
    model.load_state_dict(best_params)
#     print(model.state_dict())
    model.eval()
    model.cpu()

    print('Saving model to drive...', end='')
    model.save_to_drive()
    print('done.')

    print('Saving training logs...', end='')
    np_logs = np.stack([np.array(item) for item in logs.values()], axis=0)
    if not os.path.exists(loss_save_dir):
        os.makedirs(loss_save_dir)
    np.save(loss_save_dir+model.DEFAULT_SAVED_NAME, np_logs)
    print('done.')


def train_ae(model, train_loader, test_loader, device, loss_save_dir, M=1, num_epochs=1000):

    optimizer = Adam(model.parameters(), lr=1e-2)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    train_unsupervised(model, optimizer, scheduler, train_loader, test_loader, device, loss_save_dir,
                       M=M, num_epochs=num_epochs)


if __name__ == '__main__':
    # load dataset
    trainset = TreeData(data_folder="./data/Tree_2000_64_batch5.pickle", train=True, split=0.8, n_feature=8)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    testset = TreeData(data_folder="./data/Tree_2000_64_batch5.pickle", train=False, split=0.8, n_feature=8)
    # python iterable over a dataset
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    # train model

    # Sets the current device
    torch.cuda.set_device(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # log file
    loss_save_dir = './log/'
    writer = SummaryWriter('tree_batch8')
    # create an instance of AE() class
    model = AE(device, weight=1, save_name='tree''_batch', n_feature=8)
    model.to(device)
    # create an instance of train_ae() class
    train_ae(model, train_loader, test_loader, device, loss_save_dir, num_epochs=1000, M=1)
    # export class scalars as json file
    writer.export_scalars_to_json("./tree_batch.json")
    writer.close()

    # test
    X, I_list, Feature, Node_is_leaf = next(iter(test_loader))
    X = X.squeeze(0)
    # remove single-dimensional entries from the shape of an array numpy.squeeze()
    Feature = Feature.squeeze(0)
    Node_is_leaf = Node_is_leaf.squeeze(0)
    X = X.float()
    # call encode() from AE()
    Feature_New = model.encode(X, Feature, I_list)
    X_New = torch.zeros(X.shape)
    # call decode() from AE()
    X_New, Feature, Loss, Loss_P, Loss_Leaf, Num = model.decode(X, Node_is_leaf, X_New, Feature_New, I_list)

    X = X.detach().numpy()
    X_New = X_New.detach().numpy()

    # visualzation
    # plot the result of first tree
    X_tree0 = X[:127]
    X_tree0_r = X_New[:127]
    min_xy_list, s_list, min_xy_list2, s_list2 = get_minxy_s_list(X_tree0, X_tree0_r, I_list)

    plot_tree(X_tree0, I_list, min_xy_list)
    plt.savefig('./log/tree1.png')
    plt.show()

    X_tree0_r2 = np.hstack((X_tree0_r[:,:2], s_list2))
    plot_tree(X_tree0_r2, I_list, min_xy_list2)
    plt.savefig('./log/tree1_r.png')
    plt.show()

    # plot the position result of different level
    data_list = []
    txt_list = []
    for i, Ii in enumerate(I_list):
        Ii = Ii.squeeze(0).detach().numpy()
        node_index = (Ii[Ii[:,2] < 127][:,2]).astype('int32')
        data_list.append(X[node_index,:2])
        txt_list.append(node_index)
        data_list.append(X_New[node_index,:2])
        txt_list.append(node_index)

    plot_samples(data_list, txt_list, len(I_list), 2, save=True, savename='./log/tree_batch2.png')


