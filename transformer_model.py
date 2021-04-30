import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

MODELS_EXT = '.dms'

class SaveableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save_to_drive(self, name=None):
        if not os.path.exists(self.MODELS_DIR):
            os.makedirs(self.MODELS_DIR)
        name = name if name is not None else self.DEFAULT_SAVED_NAME
        torch.save(self.state_dict(), os.path.join(self.MODELS_DIR, name+MODELS_EXT))

    def load_from_drive(model, name=None, model_dir=None, **kwargs):
        name = name if name is not None else model.DEFAULT_SAVED_NAME
        loaded = model(**kwargs)
        loaded.load_state_dict(torch.load(os.path.join(model_dir, name+MODELS_EXT)))
        loaded.eval()
        return loaded

class TreeData(Dataset):
    def __init__(self, data_folder='./Tree_2000_64_batch5.pickle', train = True, split = 0.8, n_feature = 16, num_box=8, batch_size=5):
        self.data_folder = data_folder
        self.n_feature = n_feature
        self.num_box = num_box
        self.batch_size = batch_size
        node_list, I_List = pickle.load(open(data_folder, "rb" ))
        num = len(node_list)
        if train:
            self.node_list = node_list[:int(num * split)]
            self.I_List = I_List[:int(num * split)]
        else:
            self.node_list = node_list[int(num * split) : num]
            self.I_List = I_List[int(num * split) : num]
     
    def __getitem__(self, idx):   
        if torch.is_tensor(idx):
            idx = idx.tolist()      
        node_xys = self.node_list[idx]
        I_list = self.I_List[idx]
        I_list = [t.astype('int64') for t in I_list ]
        node_fea = torch.zeros(node_xys.shape[0], self.n_feature)
        # node_fea = np.hstack((self.node_list[idx][:,:2],self.node_list[idx][:,3:6]))
        # node_fea = self.node_list[idx][:,3:6]
        node_is_leaf = torch.FloatTensor([self.num_box * [1] + (self.num_box-1) * [0]] * self.batch_size).view(-1,1)
        return node_xys, I_list, node_fea, node_is_leaf
    
    def __len__(self):
        return len(self.node_list)

class TFEncoder(nn.Module):
    def __init__(self):
        super(TFEncoder, self).__init__()
        self.tf = nn.Transformer(d_model=5,nhead=1,num_encoder_layers=1,num_decoder_layers=1)

    def forward(self, X_left, X_right):

        lf = self.tf(X_left,X_left)
        rf = self.tf(X_right,X_right)
        #out_feature = torch.cat((lf,rf), 1)

        return lf, rf

class AE(SaveableModule):
    def __init__(self, device, leaf_loss = False, weight_type=0, weight_leaf=1, model_folder='log', save_name='ae', n_feature = 16, encode_in_channel = 6, decode_in_channel = 19):
        super(AE, self).__init__()

        self.n_feature = n_feature
        self.weight_leaf = weight_leaf
        self.weight_type = weight_type
        self.leaf_loss = leaf_loss
        self.MODELS_DIR = model_folder
        self.DEFAULT_SAVED_NAME =  save_name
        self.device = device
        self.Transformer = TFEncoder()

    def transformer(self, X, Node_is_leaf, I_list):

        num_I = len(I_list)  # nlevel
        X_r = X.clone()
        X_r = X_r.squeeze(0)
        X_ab_xy = X.clone()
        X_ab_xy_r = X.clone()
        Loss_P = 0.0
        Loss_Leaf = 0.0
        num = 0
        left_check = []
        right_check = []
        # Feature = torch.empty(3150, 5)

        for i in range(num_I):
            I = I_list[num_I-1-i].squeeze(0)
            left_p = torch.reshape(X_r[I[:, 0]], (len(I), 1, 5))
            right_p = torch.reshape(X_r[I[:, 1]], (len(I), 1, 5))

            out_left, out_right = self.Transformer(left_p, right_p)
            #out_left, out_right = torch.chunk(out, 2, 1)
            out_left = torch.reshape(out_left, (len(I), 5))
            out_right = torch.reshape(out_right, (len(I), 5))

            # Feature[I[:, 0]] = out_left
            # Feature[I[:, 1]] = out_right

            p_left = X_ab_xy[I[:,0]]  # (ni, 6)
            p_right = X_ab_xy[I[:,1]]  # (ni, 6)
            # leaf_left = Node_is_leaf[I[:,0]]  # (ni, 1)
            # leaf_right = Node_is_leaf[I[:,1]]  # (ni, 1)

            weight = 1

            if(self.weight_type==1):
                weight = (2**(num_I-i-1))
            
            if(self.weight_type==2):
                weight = (2*(num_I-i))

            left_re_loss = self.cal_distance_re(p_left, out_left)
            right_re_loss = self.cal_distance_re(p_right, out_right)
            # print(self.weight_type, weight)
            Loss_P = Loss_P + (left_re_loss + right_re_loss) * weight

            # calculate leaf loss
            # if(self.leaf_loss):
            #     left_leaf_loss = self.cal_leaf_loss(leaf_left, left_isleaf)
            #     right_leaf_loss = self.cal_leaf_loss(leaf_right, right_isleaf)
            #     Loss_Leaf =  Loss_Leaf + (left_leaf_loss + right_leaf_loss) * weight
            
            left_check.append(left_re_loss)
            right_check.append(right_re_loss)

            # get GroundTruth absolute xy
            left_xy_new = X_ab_xy[I[:,0],:2] * X_ab_xy[I[:,2],2:4] + X_ab_xy[I[:,2],:2]
            X_ab_xy[I[:,0],:2] = left_xy_new
            left_wh_new = X_ab_xy[I[:,0],2:4] * X_ab_xy[I[:,2],2:4]
            X_ab_xy[I[:,0],2:4] = left_wh_new
            left_a_new = X_ab_xy[I[:,0],4] + X_ab_xy[I[:,2],4]
            X_ab_xy[I[:,0],4] = left_a_new

            right_xy_new = X_ab_xy[I[:,1],:2] * X_ab_xy[I[:,2],2:4] + X_ab_xy[I[:,2],:2]
            X_ab_xy[I[:,1],:2] = right_xy_new
            right_wh_new = X_ab_xy[I[:,1],2:4] * X_ab_xy[I[:,2],2:4]
            X_ab_xy[I[:,1],2:4] = right_wh_new
            right_a_new = X_ab_xy[I[:,1],4] + X_ab_xy[I[:,2],4]
            X_ab_xy[I[:,1],4] = right_a_new

            # get Reconstruction absolute xy
            left_xy_new = out_left[:,:2] * X_ab_xy_r[I[:,2],2:4] + X_ab_xy_r[I[:,2],:2]
            left_wh_new = out_left[:,2:4] * X_ab_xy_r[I[:,2],2:4]
            left_a_new = out_left[:,4] + X_ab_xy_r[I[:,2],4]
            X_ab_xy_r[I[:,0],:2] = left_xy_new
            X_ab_xy_r[I[:,0],2:4] = left_wh_new
            X_ab_xy_r[I[:,0],4] = left_a_new

            right_xy_new = out_right[:,:2] * X_ab_xy_r[I[:,2],2:4] + X_ab_xy_r[I[:,2],:2]
            right_wh_new = out_right[:,2:4] * X_ab_xy_r[I[:,2],2:4]
            right_a_new = out_right[:,4] + X_ab_xy_r[I[:,2],4]
            X_ab_xy_r[I[:,1],:2] = right_xy_new
            X_ab_xy_r[I[:,1],2:4] = right_wh_new
            X_ab_xy_r[I[:,1],4] = right_a_new

        return X_ab_xy, X_ab_xy_r, Loss_P, num_I, left_check, right_check
        
    def forward(self, X, I_list, Node_is_leaf):

        X = X.squeeze(0)
        X_ab_xy, X_ab_xy_r, Loss_P, Num, left_check, right_check = self.transformer(X, Node_is_leaf, I_list)

        Loss = Loss_P
        Loss = Loss.requires_grad_()

        return Loss, left_check, right_check
    
    def cal_distance_re(self, q, p):
        '''
        Input:
            p, q: position of a node  n*3
        Output:
            dis: The distance of two nodes
        '''         
        # dis_xy =  F.mse_loss(p[:,:2], q[:,:2], reduction='mean')
        # dis_wh =  F.mse_loss(p[:,2:4], q[:,2:4], reduction='mean')
        # dis_a =  F.l1_loss(p[:,4:], q[:,4:], reduction='mean')
        # dis = dis_xy + dis_wh + dis_a
        # dis = dis_xy + dis_wh
        dis =  F.l1_loss(p, q, reduction='mean')
        
        return dis

    def cal_leaf_loss(self, left_is_leaf_gt, left_is_leaf_rec):
        '''
        Input:
            p, q: position of a node  n*3
        Output:
            dis: The distance of two nodes
        '''         
        leaf_loss =  F.binary_cross_entropy(left_is_leaf_rec, left_is_leaf_gt, reduction='mean')
        return leaf_loss

    def inference(self, P, F, n, out=[]):
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
        
        left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf = self.decoder(F, P)
        out.append([left_P, right_P, P])

        left_isleaf = torch.round(left_isleaf)
        right_isleaf = torch.round(right_isleaf)

        if(not left_isleaf):
            self.inference(left_P, left_featrue, n-1, out)
        if(not right_isleaf):
            self.inference(right_P, right_featrue, n-1, out)
        else:
            return out


    def loss_on_loader(self, loader, device):
        # calculate loss on all data
        total = 0.0
        total_ab = 0.0
        total_p = 0.0
        total_leaf = 0.0
        num = 0

        train_loss_left_check = np.zeros((20,1))
        train_loss_right_check = np.zeros((20,1))
        
        with torch.no_grad():
            for i,(node_xys, I_list, node_fea, node_is_leaf) in enumerate(loader, 0):
                node_xys = node_xys.to(device)
                node_xys = node_xys.float()
                I_list = [t.to(device) for t in I_list ]
                node_is_leaf = node_is_leaf.to(device)
                loss, left_check, right_check = self.forward(node_xys,I_list,node_is_leaf)
                
                for i,ITEM in enumerate(left_check):
                    train_loss_left_check[i] += ITEM.item()
                for i,ITEM in enumerate(right_check):
                    train_loss_right_check[i] += ITEM.item()

                total += loss
                num += 1
        return total/num, train_loss_left_check/num, train_loss_right_check/num

if __name__  == '__main__':

    Dataset = TreeData(root_dir)
    train_loader = torch.utils.data.DataLoader(Dataset, batch_size=1)
    node_xys, I_list, node_fea, node_is_leaf = next(iter(train_loader))
    model = AE()
    out = model(node_xys, node_fea, I_list,  node_is_leaf)