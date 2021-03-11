#!/usr/bin/env python
# coding: utf-8

import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import shapely
from shapely.geometry import Polygon
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import leaves_list


data_folder = 'Data/Manhattan_polygon2_45847.pickle'
with open(data_folder, 'rb') as filename:
    polygon_list = pickle.load(filename)

center_mean = np.zeros((len(polygon_list), 2))
for i in range(len(polygon_list)):
    center_mean[i] = np.mean(polygon_list[i], axis=0)[:2]
    
index = np.argsort(center_mean[:,0] + center_mean[:,1])
center_mean_sort = center_mean[index]
len(center_mean_sort)

# get Point Data, Tree Depth, Linkage Matrix
train_index = random.sample(range(len(center_mean_sort)), 2000)
center_list = []
depth_list = []
Z_list = []

N = 64
for i in range(len(train_index)):
    t = train_index[i]

    index_sort = np.argsort(np.sum(abs(center_mean_sort - center_mean_sort[t]),1))
    near_poly = index_sort[:N]
    center = center_mean_sort[near_poly]
    center2 = center - center[0]
    center2 =(center2 -  np.min(center2,axis=0))/(np.max(center2,axis=0) - np.min(center2,axis=0))
    Z = hierarchy.linkage(center2, 'ward')
    depth = len(np.unique(Z[:,3]))

    Z_list.append(Z)
    center_list.append(center2)
    depth_list.append(depth)
    
# get the Depth of Depest Tree
max_len = max(depth_list)


Node_XYS = []
I_List = []
for n in range(len(Z_list)):
    linkage = Z_list[n]
    node = center_list[n]
    print(n)
    for i in range(len(linkage)):
        new_node = np.mean(node[linkage[i,0:2].astype('int8')],axis=0)
        node = np.vstack((node,new_node))
        
    label = np.unique(linkage[:,3])

    child_list = [[] for _ in range(len(node))]
    s_list = np.zeros(len(node))
    I_list = []
    for k in range(len(label)):
#         print(k)
        index = (linkage[:,3] == label[k])
        child_node = linkage[index,:2]
        father_node = np.arange(len(linkage))[index]+len(leaf_node)
        father_node = father_node.reshape(len(father_node),1)
        I = np.hstack((child_node,father_node))
        I_list.append(I)
        
        for i,idx in enumerate(father_node):
            tmp_child = []
            for node_id in child_node[i]:
                if(len(child_list[int(node_id)])>0):
                    for child in child_list[int(node_id)]:
                        tmp_child.append(child)
                else: 
                    tmp_child.append(int(node_id))
            child_xy = node[tmp_child]
            s = max(np.max(child_xy, axis=0) - np.min(child_xy, axis=0))
            child_list[int(idx)] = tmp_child
            s_list[int(idx)] = s
            
    s_list = s_list.reshape(len(s_list),1)
    node_s = np.hstack((node, s_list))
    print(I_list)
    Node_XYS.append(node_s)
    I_List.append(I_list)


Batch_size = 5


batch_xys = []
batch_I = []
for i in range(len(Node_XYS)//5):
    for j in range(Batch_size):

        idx= i*Batch_size+j

        if(j == 0):
            tmp_xys = Node_XYS[idx]
            tmp_I = I_List[idx]
        else:
            tmp_xys = np.vstack((tmp_xys, Node_XYS[idx]))
            
            max_len = max(len(tmp_I), len(I_List[idx]))
            
            for k in range(max_len):
                if((k<len(tmp_I)) and (k<len(I_List[idx]))):
                    new_I = I_List[idx][k]+j*len(Node_XYS[idx])
                    tmp_I[k] = np.vstack((tmp_I[k],new_I))
                else:
                    if(k<len(I_List[idx])):
                        new_I = I_List[idx][k]+j*len(Node_XYS[idx])
                        tmp_I.append(new_I)
        
    batch_xys.append(tmp_xys)
    batch_I.append(tmp_I)


filename = 'Tree_AE/Tree_2000_64_batch5.pickle'
pfile = open(filename,'wb')
pickle.dump((batch_xys, batch_I), pfile, protocol=2)
pfile.close()

