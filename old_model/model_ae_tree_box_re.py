from torch.utils.data import Dataset
import torch 
import pickle
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import numpy as np

MODELS_EXT = '.dms'

def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    return li

def get_MLP_layers(dims, doLastRelu=False):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i]))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

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
    def __init__(self, data_folder='./Tree_2000_64_batch5.pickle', train = True, split = 0.8, n_feature = 16):
        self.data_folder = data_folder
        self.n_feature = n_feature
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
        node_is_leaf = torch.FloatTensor([64 * [1] + 63 * [0]] * 5).view(-1,1)
        return node_xys, I_list, node_fea, node_is_leaf
    
    def __len__(self):
        return len(self.node_list)

class Encoder(nn.Module):
    def __init__(self, n_feature = 16, in_channel = 16+3):
        super(Encoder, self).__init__()
        self.n_feature = n_feature
        out_channel = n_feature
        self.W = get_MLP_layers((in_channel, n_feature, n_feature, n_feature))
        # self.b = get_MLP_layers((in_channel, n_feature, n_feature, n_feature))

    def forward(self, X_left, X_right, Feature_left, Feature_right):
        '''
        Input:
            X_left: Position of Left Child Node  n*3
            X_right: Position of Right Child Node  n*3
            Feature_left: Feature of Left Child Node  n*d
            Feature_right: Feature of Right Child Node  n*d
        Output:
            out_feature: Feature of Father Node n*d
        '''
        input_left = torch.cat((X_left, Feature_left),1)  # (n, 6+d)
        input_right = torch.cat((X_right, Feature_right),1)  # (n, 6+d)
        out_feature = self.W(input_left)  + self.W(input_right) # (n, d)
        
        return out_feature
    
class Decoder(nn.Module):
    def __init__(self, n_feature = 16, in_channel = 19):
        super(Decoder, self).__init__()
        self.n_feature = n_feature
        out_channel = (n_feature + 6 + 1) * 2
        self.M = get_MLP_layers((in_channel, n_feature, n_feature*2, out_channel))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, Feature_father, P_father):
        '''
        Input:
            Feature_father: Feature of Father Node  n*d
            P_father: Position of Father Node  n*3
        Output:
            left_featrue: Feature of Left Child Node  n*d
            left_P: Position of Left Child Node  n*3
            left_isleaf: Left Child Node is leaf node or not (True or False)  n*1
            right_featrue: Feature of Right Child Node  n*d
            right_P: Position of Right Child Node  n*3
            right_isleaf: Right Child Node is leaf node or not (True or False)  n*1
        ''' 
        input_father = torch.cat((Feature_father, P_father),1)  # (n, d+3)
        out_father = self.M(input_father)  # (n, (d+3+1)*2)
        
        left_featrue = out_father[:,:self.n_feature]  # (n, d)
        left_P_xy = self.tanh(out_father[:,self.n_feature:(self.n_feature + 2)])  # (n, 2)
        left_P_s_wha = self.sigmoid(out_father[:,(self.n_feature + 2):(self.n_feature + 6)])  # (n, 4)
        left_P = torch.cat((left_P_xy, left_P_s_wha), 1)  # (n, 6)
        left_isleaf = self.sigmoid(out_father[:,(self.n_feature + 6):(self.n_feature + 7)])  # (n, 1)
        
        right_featrue = out_father[:,(self.n_feature + 7):(self.n_feature * 2 + 7)]  # (n, d)
        right_P_xy = self.tanh(out_father[:,(self.n_feature * 2 + 7):(self.n_feature * 2 + 9)])  # (n, 2)
        right_P_s_wha = self.sigmoid(out_father[:,(self.n_feature * 2 + 9):(self.n_feature * 2 + 13)])  # (n, 4)
        right_P =  torch.cat((right_P_xy, right_P_s_wha), 1)  # (n, 6)
        right_isleaf = self.sigmoid(out_father[:,(self.n_feature * 2 + 13):])  # (n, 1)
        
        return left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf

class AE(SaveableModule):
    def __init__(self, device, weight=1, model_folder='log', save_name='ae', n_feature = 16, encode_in_channel = 6, decode_in_channel = 19):
        super(AE, self).__init__()

        self.n_feature = n_feature
        self.encoder = Encoder(n_feature, n_feature+6)
        self.decoder = Decoder(n_feature, n_feature+6)
        # self.G = get_MLP_layers((2, n_feature//4, n_feature//2, n_feature))
        self.weight = weight
        self.MODELS_DIR = model_folder
        self.DEFAULT_SAVED_NAME =  save_name
        self.device = device

    def encode(self, X, Feature, I_list):
        '''
        Input:
            X: Positions of ALL Tree Nodes  n*3
            Feature: Features of ALL Tree Nodes  n*d
            I_list: The Index Matrix of merge Nodes  nlevel*B*ni*3
        Output:
            Feature: Encoded Features (updates on input Features)  B*n*d
        ''' 
        # Feature[:64] = self.G(X[:64,:2])
        Feature_New = Feature.clone()
        for item in I_list:
            I = item.squeeze(0)  # (ni, 3)
            left_p = X[I[:,0]]  # (ni, 6) 
            right_p = X[I[:,1]]  # (ni, 6) 
            left_f = Feature[I[:,0]]  # (ni, d) 
            right_f = Feature[I[:,1]]  # (ni, d) 
            out = self.encoder(left_p, right_p, left_f, right_f)  # (ni, d) 
            Feature_New[I[:,2]] = out
        return Feature_New
    
    def decode(self, X, Node_is_leaf, Feature_New, I_list):
        '''
        Input:
            X: Positions of ALL Tree Nodes  n*3
            Node_is_leaf: Leaf marks of ALL Tree Nodes  n*1
            Feature: Encoded Features of ALL Tree Nodes  n*d
            I_list: The Index Matrix of merge Nodes  nlevel*B*ni*3
        Output:
            X_New: Decoded Positions of ALL Tree Nodes  n*3
            Feature: Decoded Features (updates on input Features)  n*d
            Loss: Sum Loss of ALL Father Nodes
            num: Number of Father Nodes
        ''' 
        num_I = len(I_list)  # nlevel
        X_ab_xy = X.clone()
        X_ab_xy_r = X.clone()
        Feature = Feature_New.clone()
        Loss_P = 0.0
        Loss_Leaf = 0.0
        num = 0
        for i in range(num_I):
            I = I_list[num_I-1-i].squeeze(0)  # (n, ni)

            p_left = X[I[:,0]]  # (ni, 6)
            p_right = X[I[:,1]]  # (ni, 6)
            leaf_left = Node_is_leaf[I[:,0]]  # (ni, 1)
            leaf_right = Node_is_leaf[I[:,1]]  # (ni, 1)

            p_father = X[I[:,2]]  # (ni, 6)
            f_father = Feature[I[:,2]]  # (ni, d)
            
            # (n, d), (n, 6), (n, 1), (n, d), (n, 6), (n, 1)
            left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf = self.decoder(f_father, p_father)

            # assume the decoded leaf node is leaf node
            out_p_left = left_P.clone()
            out_f_left = left_featrue.clone()
            out_i_left = left_isleaf.clone()
            out_p_right = right_P.clone()
            out_f_right = right_featrue.clone()
            out_i_right = right_isleaf.clone()

            # calculate loss
            position_loss, l_r_index = self.get_Position_loss(p_left, left_P, p_right, right_P)  # (ni, 1) (ni, 1)
            leaf_loss = self.get_Binary_loss(leaf_left, left_isleaf, leaf_right, right_isleaf) # (ni, 1)
            position_loss = torch.mean(position_loss) #* (num_I-i) #self.weight 
            leaf_loss = torch.mean(leaf_loss) 

            num += 1
            Loss_P += position_loss
            Loss_Leaf += leaf_loss

            # swap the decoded leaf node and decoded right node according to loss 
            swap_index = torch.arange(len(l_r_index))[l_r_index==1]
            out_p_left[swap_index] = right_P[swap_index]
            out_f_left[swap_index] = right_featrue[swap_index]
            out_i_left[swap_index] = right_isleaf[swap_index]
            out_p_right[swap_index] = left_P[swap_index]
            out_f_right[swap_index] = left_featrue[swap_index]
            out_i_right[swap_index] = left_isleaf[swap_index]
                    
            # update decoded Feature        
            Feature[I[:,0]] = out_f_left
            Feature[I[:,1]] = out_f_right

            # get GroundTruth absolute xy
            left_xy_new = X_ab_xy[I[:,0],:2] * X_ab_xy[I[:,2],2:3] + X_ab_xy[I[:,2],:2]
            X_ab_xy[I[:,0],:2] = left_xy_new
            left_wh_new = X_ab_xy[I[:,0],3:5] * X_ab_xy[I[:,2],3:5]
            X_ab_xy[I[:,0],3:5] = left_wh_new
            right_xy_new = X_ab_xy[I[:,1],:2] * X_ab_xy[I[:,2],2:3] + X_ab_xy[I[:,2],:2]
            X_ab_xy[I[:,1],:2] = right_xy_new
            right_wh_new = X_ab_xy[I[:,1],3:5] * X_ab_xy[I[:,2],3:5]
            X_ab_xy[I[:,1],3:5] = right_wh_new

            # get Reconstruction absolute xy
            left_xy_new = out_p_left[:,:2] * X_ab_xy_r[I[:,2],2:3] + X_ab_xy_r[I[:,2],:2]
            left_wh_new = out_p_left[:,3:5] * X_ab_xy_r[I[:,2],3:5]
            X_ab_xy_r[I[:,0],:2] = left_xy_new
            X_ab_xy_r[I[:,0],2] = out_p_left[:,2]
            X_ab_xy_r[I[:,0],3:5] = left_wh_new
            X_ab_xy_r[I[:,0],5] = out_p_left[:,5]

            right_xy_new = out_p_right[:,:2] * X_ab_xy_r[I[:,2],2:3] + X_ab_xy_r[I[:,2],:2]
            right_wh_new = out_p_right[:,3:5] * X_ab_xy_r[I[:,2],3:5]
            X_ab_xy_r[I[:,1],:2] = right_xy_new
            X_ab_xy_r[I[:,1],2] = out_p_right[:,2]
            X_ab_xy_r[I[:,1],3:5] = right_wh_new
            X_ab_xy_r[I[:,1],5] = out_p_right[:,5]

        return X_ab_xy, X_ab_xy_r, Feature, Loss_P, Loss_Leaf, num
        
    def forward(self, X, Feature, I_list, Node_is_leaf):
        '''
        Input:
            X: Positions of ALL Tree Nodes  B*n*3
            Feature: Features of ALL Tree Nodes  B*n*d
            I_list: The Index Matrix of merge Nodes  nlevel*B*ni*3
            Node_is_leaf: Leaf marks of ALL Tree Nodes  B*n*1
        Output:
            Loss: Sum Loss of ALL Father Nodes
            num: Number of Father Nodes
        ''' 
        X = X.squeeze(0)  # (n, 3)
        Feature = Feature.squeeze(0)  # (n, d)
        Node_is_leaf = Node_is_leaf.squeeze(0)  # (n, 1)
       
        Feature_New = self.encode(X, Feature, I_list)  # (n, d)
        # (n, 6), (n, 6), (n, d), (1), (1)
        X_ab_xy, X_ab_xy_r, Feature_r, Loss_P, Loss_Leaf, Num = self.decode(X, Node_is_leaf, Feature_New, I_list)
        # print(X_ab_xy, X_ab_xy_r)
        Loss_ab = self.cal_distance_ab(X_ab_xy, X_ab_xy_r)
        # Loss_fea = self.cal_distance_fea(Feature_New, Feature_r)
        # Loss_fea = 0
        Loss_P = Loss_P / Num
        Loss_Leaf = Loss_Leaf / Num
        Loss = Loss_ab + Loss_P + Loss_Leaf
        # print(Loss, Loss_ab , Loss_P , Loss_Leaf)
        # Loss = Loss / Num
        Loss = Loss.requires_grad_()
        return Loss, Loss_ab, Loss_P, Loss_Leaf
    
    def cal_distance(self, p, q):
        '''
        Input:
            p, q: position of a node  n*3
        Output:
            dis: The distance of two nodes
        ''' 
        px, py, ps = p[:,0], p[:,1], p[:,2]
        qx, qy, qs = q[:,0], q[:,1], q[:,2]
        
        lx = torch.max( px + ps/2, qx + qs/2) - torch.min( px - ps/2, qx - qs/2)
        ly = torch.max( py + ps/2, qy + qs/2) - torch.min( py - ps/2, qy - qs/2)
        dis = torch.abs(ps * ps + ps * ps - 2 * lx * ly)
        return dis

    def cal_distance2(self, q, p, details = False):
        '''
        Input:
            p, q: position of a node  n*3
        Output:
            dis: The distance of two nodes
        '''         
        # px, py, ps = p[:,0], p[:,1], p[:,2]
        # qx, qy, qs = q[:,0], q[:,1], q[:,2]
        # dis_xy = (((px - qx) * (px - qx)) + ((py - qy) * (py - qy)))/2
        # dis_s = torch.abs(ps - qs)
        # dis = dis_xy + dis_s
        # dis = (((px - qx) * (px - qx)) + ((py - qy) * (py - qy)) + ((ps - qs) * (ps - qs)))/3
        
        # dis_xy =  F.mse_loss(p[:,:2], q[:,:2], reduction='none')
        # dis_xy = torch.sum(dis_xy, 1)
        # dis_s =  F.l1_loss(p[:,2:], q[:,2:], reduction='none')
        # dis_s = torch.sum(dis_s, 1)
        # dis = dis_xy + dis_s
        dis_xy =  F.mse_loss(p[:,:2], q[:,:2], reduction='none')
        dis_xy = torch.sum(dis_xy, 1)
        dis_s_wha =  F.l1_loss(p[:,2:], q[:,2:], reduction='none')
        dis_s_wha = torch.sum(dis_s_wha, 1)
        dis = dis_xy + dis_s_wha
        return dis

    def cal_distance_ab(self, q, p):
        '''
        Input:
            p, q: position of a node  n*2
        Output:
            dis: The distance of two nodes
        ''' 
        dis_xy = F.mse_loss(p[:,:2], q[:,:2], reduction='mean')
        dis_s_wha =  F.l1_loss(p[:,2:], q[:,2:], reduction='mean')
        dis = dis_xy + dis_s_wha
        return dis

    def cal_distance_fea(self, q, p):
        '''
        Input:
            p, q: position of a node  n*2
        Output:
            dis: The distance of two nodes
        ''' 
        # dis_xy =  F.mse_loss(p[:,:2], q[:,:2], reduction='mean')
        # dis_wha =  F.l1_loss(p[:,2:], q[:,2:], reduction='mean')
        # dis_fea = dis_xy + dis_wha
        dis_fea =  F.l1_loss(p, q, reduction='mean')
        return dis_fea
    
    def get_Binary_loss(self, left_is_leaf, left_is_leaf2, right_is_leaf, right_is_leaf2):

        left_loss =  F.binary_cross_entropy(left_is_leaf2, left_is_leaf, reduction='none')
        right_loss = F.binary_cross_entropy(right_is_leaf2, right_is_leaf, reduction='none')
        return left_loss + right_loss
        
    def get_Position_loss(self, leaf_p, leaf_p2, right_p, right_p2):
        # l_l_r_r = self.cal_distance(leaf_p, leaf_p2) + self.cal_distance(right_p, right_p2)
        # l_r_l_r = self.cal_distance(leaf_p, right_p2) + self.cal_distance(right_p, leaf_p2)
                
        l_l_r_r = self.cal_distance2(leaf_p, leaf_p2) + self.cal_distance2(right_p, right_p2)
        l_r_l_r = self.cal_distance2(leaf_p, right_p2) + self.cal_distance2(right_p, leaf_p2)
        loss, index = torch.min(torch.stack((l_l_r_r, l_r_l_r),0),0)
        
        return loss, index

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
        with torch.no_grad():
            for i,(node_xys, I_list, node_fea, node_is_leaf) in enumerate(loader, 0):
                node_xys = node_xys.to(device)
                node_xys = node_xys.float()
                I_list = [t.to(device) for t in I_list ]
                node_fea = node_fea.to(device)
                node_fea = node_fea.float()
                node_is_leaf = node_is_leaf.to(device)
                loss, loss_ab, loss_p, loss_leaf = self.forward(node_xys,node_fea,I_list,node_is_leaf)
                
                total += loss
                total_ab += loss_ab
                total_p += loss_p
                total_leaf += loss_leaf
                num += 1
        return total/num, total_ab/num, total_p/num, total_leaf/num

if __name__  == '__main__':

    Dataset = TreeData(root_dir)
    train_loader = torch.utils.data.DataLoader(Dataset, batch_size=1)
    node_xys, I_list, node_fea, node_is_leaf = next(iter(train_loader))
    model = AE()
    out = model(node_xys, node_fea, I_list,  node_is_leaf)

