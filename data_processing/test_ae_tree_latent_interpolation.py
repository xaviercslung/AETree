
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

torch.set_printoptions(sci_mode=False)


from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

    
def rotate_xy_2(p, sin, cos, center):
    x_ = (p[:,0:1]-center[:,0:1])*cos-(p[:,1:2]-center[:,1:2])*sin+center[:,0:1]
    y_ = (p[:,0:1]-center[:,0:1])*sin+(p[:,1:2]-center[:,1:2])*cos+center[:,1:2]
#     print(((p[:,0:1]-center[:,0:1])*cos).shape, cos.shape, x_.shape)
    return np.hstack((x_, y_))

def get_box_2(P, F):
    ld = np.hstack((P[:,0:1]-F[:,0:1]/2, P[:,1:2]-F[:,1:2]/2))
    rd = np.hstack((P[:,0:1]+F[:,0:1]/2, P[:,1:2]-F[:,1:2]/2))
    ru = np.hstack((P[:,0:1]+F[:,0:1]/2, P[:,1:2]+F[:,1:2]/2))
    lu = np.hstack((P[:,0:1]-F[:,0:1]/2, P[:,1:2]+F[:,1:2]/2))
    # box = np.hstack((ld, rd, ru, lu)).reshape(len(P), -1, 2)
    sinO = np.sin(F[:,2:3])
    cosO = np.cos(F[:,2:3])

    ld_r = rotate_xy_2(ld, sinO, cosO, P)
    rd_r = rotate_xy_2(rd, sinO, cosO, P)
    ru_r = rotate_xy_2(ru, sinO, cosO, P)
    lu_r = rotate_xy_2(lu, sinO, cosO, P)
    if(len(P)>0):
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
    
    X, Y= pc[:, 0], pc[:,1]
    plt.plot(X, Y, c=color)
    plt.plot([X[-1],X[0]], [Y[-1],Y[0]], c=color)
#     if(txt):
#         plt.annotate(txt, center, size=8)
#     plt.scatter(center[0], center[1], c='b')
#     n = np.arange(len(pc))
#     for i,txt in enumerate(n):
#         plt.annotate(txt,(pc[i,0],pc[i,1]),size=8)
        
   
    plt.axis('off')
#     plt.show()

def draw_box(box, txt, center, color):
#     print(box.shape)
    for i, p in enumerate(box):
#         print(p.shape)
        c = 'r' if color[i] else 'b'
        draw_polygon_c(p, txt[i], center[i], c)

def plot_boxes(samples, label, center, color, n, m, save=False, savename='pclouds'):
    fig = plt.figure(figsize=(5*m,5*n))
    fig.set_tight_layout(True)
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            ax = fig.add_subplot(n, m, idx+1)
#             print(idx)
            draw_box(samples[idx], label[idx], center[idx], color[idx])  
    if save:
        plt.savefig(savename)

    plt.show()



from model_ae_tree_box_ab2_new_re_weight_lstm_print import *

trainset = TreeData(data_folder="../Tree_2000_64_batch5_box_re_new2_32_45000_batch50_2.pickle", train=True, split=0.9, n_feature=512, num_box=32, batch_size=50)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

testset = TreeData(data_folder="../Tree_2000_64_batch5_box_re_new2_32_45000_batch50_2.pickle", train=False, split=0.9, n_feature=512, num_box=32, batch_size=50)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# testset = TreeData(data_folder="../Tree_2000_64_batch5_box_re2.pickle", train=False, split=0.8, n_feature=128)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

torch.cuda.set_device(2)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = AE.load_from_drive(AE, name='tree_lstm_32_print_45000_time_leaf_rerun2_3_best', model_dir='./log', device=device, n_feature=512)


# In[4]:


def inference(P, F, n, out=[]):
    '''
    Input:
        P: Positions of initial node  n*3
        F: Features of initial node  n*d
        n: Max level of generated tree
    Output:
        out: Position List of generated children nodes
    ''' 
    if(n == 0):
        return out
    P_list = []
    P_list_re = []
    I_list = []
    leaf_node_list = []
    
    tmp_P = P
    tmp_F = F
    father_I = torch.zeros((1,1))
    P_list.append(P[0])
#     print(n)
    for i in range(n):
#         print('--------------',i,'---------------')
        left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf = model.decoder(tmp_F, tmp_P)
#         print(left_featrue.shape, left_P.shape, left_isleaf.shape, right_featrue.shape, right_P.shape, right_isleaf.shape)
#         print(left_P, left_isleaf, right_P, right_isleaf)
        
        left_xy_new = left_P[:,:2] * tmp_P[:,2:4] + tmp_P[:,:2]
        left_P[:,:2] = left_xy_new
        left_wh_new = left_P[:,2:4] * tmp_P[:,2:4]
        left_P[:,2:4] = left_wh_new
        left_a_new = left_P[:,4] + tmp_P[:,4]
        left_P[:,4] = left_a_new
        
        right_xy_new = right_P[:,:2] * tmp_P[:,2:4] + tmp_P[:,:2]
        right_P[:,:2] = right_xy_new
        right_wh_new = right_P[:,2:4] * tmp_P[:,2:4]
        right_P[:,2:4] = right_wh_new
        right_a_new = right_P[:,4] + tmp_P[:,4]
        right_P[:,4] = right_a_new
        
        I = np.zeros(len(left_P)*2, dtype='int32')
        
        temp_I_list = []
        left_node_index = torch.zeros((len(left_P),1))
        right_node_index = torch.zeros((len(right_P),1))
        
        for j in range(len(left_P)):
            
            P_list.append(left_P[j])
            left_index =len(P_list) -1
            P_list.append(right_P[j])
            right_index =len(P_list) -1
            
            father_index = father_I[j].detach().numpy()
            father_index = int(father_index)
            temp_I = [left_index, right_index, father_index]
            
            I[2*j] = left_index
            I[2*j+1] = right_index
            left_node_index[j] = left_index
            right_node_index[j] = right_index
            
#             print('*******I:', temp_I)
            temp_I_list.append(temp_I)
            
        if(temp_I_list):     
            I_list.append(temp_I_list) 
            
        
#         print(I)
    
        tmp_F = []
        tmp_P = []
        father_I = []
        
#         print(left_isleaf, right_isleaf)
        
        left_isleaf = torch.round(left_isleaf)[:,0]
        right_isleaf = torch.round(right_isleaf)[:,0]
        
#         print(left_isleaf, right_isleaf)
        
        tmp_F.append(left_featrue[left_isleaf==0,:])
        tmp_P.append(left_P[left_isleaf==0,:])
        father_I.append(left_node_index[left_isleaf==0,:])
        
        tmp_F.append(right_featrue[right_isleaf==0,:])
        tmp_P.append(right_P[right_isleaf==0,:])
        father_I.append(right_node_index[right_isleaf==0,:])
        
#         print('lllllll', father_I)
#         print(father_I[0].shape)
        tmp_F = torch.cat(tmp_F, 0)
        tmp_P = torch.cat(tmp_P, 0)
        father_I = torch.cat(father_I, 0)
#         print(father_I[0].shape)
        
#         print(tmp_F.shape)
#         print(tmp_P.shape)
        
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
    P = X_ab_r[node_index,:2]
    F = X_ab_r[node_index,2:]
    box = get_box_2(P,F)
    box_infer_list.append(box)
    center_infer_list.append(P)
    txt_infer_list.append(node_index)
    color_infer_list.append(color)  

    for Ii in I_list:
        Ii= np.array(Ii)
        left_idx = (Ii[:,0]).astype('int32')
        right_idx = (Ii[:,1]).astype('int32')
        father_idx = (Ii[:,2]).astype('int32')
    #     print(left_idx, right_idx, father_idx,node_index)
        node_index = list(set(node_index) -  set(father_idx) | set(left_idx) | set(right_idx))
    #     print(node_index)
        color = np.zeros(len(node_index))
        for idx in father_idx:
            color[(np.arange(len(node_index))[node_index==idx])]=1 

        if(len(node_index)>0):
            P = X_ab_r[node_index,:2]
            F = X_ab_r[node_index,2:]
    #         print(P,F)
            box = get_box_2(P,F)
    #         print(box)
            box_infer_list.append(box)
            center_infer_list.append(P)
            txt_infer_list.append(node_index)
            color_infer_list.append(color)
            n = n + 1
    return box_infer_list[-1]

def inference_final_set(P, F, n, out=[]):
    '''
    Input:
        P: Positions of initial node  n*3
        F: Features of initial node  n*d
        n: Max level of generated tree
    Output:
        out: Position List of generated children nodes
    ''' 
    if(n == 0):
        return out
    P_list = []
    P_list_re = []
    I_list = []
    leaf_node_list = []
    
    tmp_P = P
    tmp_F = F
    father_I = torch.zeros((1,1))
    P_list.append(P[0])

    for i in range(n):
        left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf = model.decoder(tmp_F, tmp_P)
        
        left_xy_new = left_P[:,:2] * tmp_P[:,2:4] + tmp_P[:,:2]
        left_P[:,:2] = left_xy_new
        left_wh_new = left_P[:,2:4] * tmp_P[:,2:4]
        left_P[:,2:4] = left_wh_new
        left_a_new = left_P[:,4] + tmp_P[:,4]
        left_P[:,4] = left_a_new
        
        right_xy_new = right_P[:,:2] * tmp_P[:,2:4] + tmp_P[:,:2]
        right_P[:,:2] = right_xy_new
        right_wh_new = right_P[:,2:4] * tmp_P[:,2:4]
        right_P[:,2:4] = right_wh_new
        right_a_new = right_P[:,4] + tmp_P[:,4]
        right_P[:,4] = right_a_new
        
        I = np.zeros(len(left_P)*2, dtype='int32')
        
        temp_I_list = []
        left_node_index = torch.zeros((len(left_P),1))
        right_node_index = torch.zeros((len(right_P),1))
        
        for j in range(len(left_P)):
            
            P_list.append(left_P[j])
            left_index =len(P_list) -1
            P_list.append(right_P[j])
            right_index =len(P_list) -1
            
            father_index = father_I[j].detach().numpy()
            father_index = int(father_index)
            temp_I = [left_index, right_index, father_index]
            
            I[2*j] = left_index
            I[2*j+1] = right_index
            left_node_index[j] = left_index
            right_node_index[j] = right_index

            temp_I_list.append(temp_I)
            
        if(temp_I_list):     
            I_list.append(temp_I_list) 
            
    
        tmp_F = []
        tmp_P = []
        father_I = []
        
        left_isleaf = torch.round(left_isleaf)[:,0]
        right_isleaf = torch.round(right_isleaf)[:,0]
        

        
        tmp_F.append(left_featrue[left_isleaf==0,:])
        tmp_P.append(left_P[left_isleaf==0,:])
        father_I.append(left_node_index[left_isleaf==0,:])
        
        tmp_F.append(right_featrue[right_isleaf==0,:])
        tmp_P.append(right_P[right_isleaf==0,:])
        father_I.append(right_node_index[right_isleaf==0,:])
        
        if(len(left_P[left_isleaf==1,:])>0):
            leaf_node_list.append(left_P[left_isleaf==1,:])
        if(len(right_P[right_isleaf==1,:])>0):
            leaf_node_list.append(right_P[right_isleaf==1,:])
        
        tmp_F = torch.cat(tmp_F, 0)
        tmp_P = torch.cat(tmp_P, 0)
        father_I = torch.cat(father_I, 0)

        
    P_list = torch.stack(P_list)
    leaf_node_list = torch.cat(leaf_node_list)
    return P_list, I_list, leaf_node_list



root_F_list = []
root_P_list = []

Batch = 50
N = 32
s = 2*N-2
e = 2*N-1

for i,(X, I_list, Feature, Node_is_leaf) in enumerate(train_loader, 0):
#     print(i)
    X = X.squeeze(0)
    Feature = Feature.squeeze(0)
    Node_is_leaf = Node_is_leaf.squeeze(0)
    X = X.float()
    Feature = Feature.float()

    Feature_New = model.encode(X, Feature, I_list)
    X_r, X_ab_xy, X_ab_xy_r, Feature_r, Loss_P, Loss_Leaf, Num, _, _ = model.decode(X, Node_is_leaf, Feature_New, I_list)

    for j in range(Batch):
        root_F = Feature_r[s+j*e:e+j*e]
        root_F_list.append(root_F)
        
        root_P = X_ab_xy_r[s+j*e:e+j*e]
        root_P_list.append(root_P)








def draw_polygon_s(pc, w, c):
    
    X, Y= pc[:, 0], pc[:,1]
    plt.plot(X, Y, linewidth=w, color=c)
    plt.plot([X[-1],X[0]], [Y[-1],Y[0]], linewidth =w, color=c)
    plt.axis('equal')
    plt.axis('off')
        
def draw_box_save(box, linewidth='3', color='mediumslateblue', name='gt'): 
    fig = plt.figure(figsize=(6,6))
    for i, p in enumerate(box):
        draw_polygon_s(p, linewidth, color)
    plt.savefig(name,bbox_inches='tight',dpi=300,pad_inches=0.0)










def get_gt_pre(lantent_gmm):
    Batch = 50
    N = 32
    s = 0
    e = 2*N-1

    gt_list = []
    pre_list = []

    for i,(X, I_list, Feature, Node_is_leaf) in enumerate(test_loader, 0):
        X = X.squeeze(0)
        Feature = Feature.squeeze(0)
        Node_is_leaf = Node_is_leaf.squeeze(0)
        X = X.float()
        Feature = Feature.float()

        Feature_New = model.encode(X, Feature, I_list)
        X_r, X_ab_xy, X_ab_xy_r, Feature_r, Loss_P, Loss_Leaf, Num, _, _ = model.decode(X, Node_is_leaf, Feature_New, I_list)

        X_ab_xy = X_ab_xy.detach().numpy()

        for j in range(Batch):
            gt_p = X_ab_xy[s+j*e:s+j*e+N]
            gt_box = get_box_2(gt_p[:,:2], gt_p[:,2:])    
            gt_list.append(gt_box)

    Box_Set_List = []
    for i in range(4500):
        P_list, I_list, Set_list = inference_final_set(root_P_list[i],lantent_gmm[i:i+1], 20)
        Set_list = Set_list.detach().numpy()
        pre_box = get_box_2(Set_list[:,:2],Set_list[:,2:])
        pre_list.append(pre_box)
    return gt_list, pre_list


def draw_polygon_c2(pc, color='b'):
    
    X, Y= pc[:, 0], pc[:,1]
    plt.plot(X, Y, c=color)
    plt.plot([X[-1],X[0]], [Y[-1],Y[0]], c=color)
    plt.axis('equal')
    
def draw_box2(box):
    for i, p in enumerate(box):
        draw_polygon_c2(p)
#     plt.show()

def plot_boxes2(samples, n, m, save=False, savename='pclouds'):
    fig = plt.figure(figsize=(5*m,5*n))
    fig.set_tight_layout(True)
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            ax = fig.add_subplot(n, m, idx+1)
            draw_box2(samples[idx])  
    if save:
        plt.savefig(savename)

    plt.show()

# ---------------inference--------------- #

lantent_ori = torch.from_numpy(latents).contiguous().float()
Box_Set_List = []
for i in range(200):
    P_list, I_list, Set_list = inference_final_set(root_P_list[i],lantent_ori[i:i+1], 20)
    Set_list = Set_list.detach().numpy()
    box_set = get_box_2(Set_list[:,:2],Set_list[:,2:])
    Box_Set_List.append(box_set)
plot_boxes2(Box_Set_List,20,10)


# ---------------GMM--------------- #
from sklearn.mixture import GaussianMixture as GMM


from sklearn.decomposition import PCA
pca = PCA(0.999, whiten=True)
data = pca.fit_transform(latents)

# data = latents

gmm = GMM(4500, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)

data_new, components = gmm.sample(4500)
labels = gmm.predict(data)
index_list = []
for i in range(4500):
#     print(i)
    index_i = np.arange(4500)[components==labels[i]]
#     if(len(index_i)>0)
    index_i_sample = np.random.choice(index_i)
    index_list.append(index_i_sample)    
latent_cond = data_new[index_list]
latent_cond = torch.from_numpy(latent_cond).contiguous().float()
gt_list, pre_list = get_gt_pre(latent_cond)

plot_boxes2(pre_list,8,8)


filename = 'Tree_AE_gt_pre_GMM_full_PCA.pickle'
pfile = open(filename,'wb')
pickle.dump((gt_list, pre_list), pfile, protocol=2)
pfile.close()

# ---------------latent interpolation--------------- #
latents = torch.cat(root_F_list, dim=0).detach().numpy()

lantent_ori = torch.from_numpy(latents).contiguous().float()

latent0 = lantent_ori[106]
latent1 = lantent_ori[59]

steps = [latent0]+[latent0 + x*(latent1-latent0) for x in torch.linspace(0.1, 0.8, steps=10)]+[latent1]
latent_steps = torch.stack(steps, dim=0)

P0 = root_P_list[96]
P1 = root_P_list[106]

p_steps = [P0]+[P0 + x*(P1-P0) for x in torch.linspace(0.1, 0.8, steps=10)]+[P1]
P_steps = torch.stack(p_steps, dim=0)

# root_P = torch.Tensor([[0.4, 0.6, 0.8, 0.9, -0.4]])
Box_Set_List = []
for i in range(12):
    P_list, I_list, Set_list = inference_final_set(P_steps[i],latent_steps[i:i+1], 20)
    Set_list = Set_list.detach().numpy()
    box_set = get_box_2(Set_list[:,:2],Set_list[:,2:])
    Box_Set_List.append(box_set)
plot_boxes2(Box_Set_List,2,6)

folder = './figure/interpolation/'
if not os.path.exists(folder):
    os.makedirs(folder)
for i in range(12):
    draw_box_save(Box_Set_List[i], name = folder+str(i)+'.png')