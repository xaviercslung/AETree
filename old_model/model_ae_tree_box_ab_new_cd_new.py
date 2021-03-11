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


def ChamfersDistance(input1, input2, p=2):

    B, N, K = input1.shape
    _, M, _ = input2.shape

    # Repeat (x,y,z) M times in a row
    input11 = input1.unsqueeze(2)           # BxNx1xK
    input11 = input11.expand(B, N, M, K)    # BxNxMxK
    # Repeat (x,y,z) N times in a column
    input22 = input2.unsqueeze(1)           # Bx1xMxK
    input22 = input22.expand(B, N, M, K)    # BxNxMxK
    # compute the distance matrix
    D = input11 - input22                   # BxNxMxK
    D = torch.norm(D, p=2, dim=3)         # BxNxM

    dist0, _ = torch.min(D, dim=1)        # BxM
    dist1, _ = torch.min(D, dim=2)        # BxN

    dist0 = torch.sum(dist0, 1)
    dist1 = torch.sum(dist1, 1)

    loss = dist0 + dist1
    loss = torch.mean(loss) 

    return loss

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
        self.W = get_MLP_layers((in_channel, n_feature, n_feature, n_feature)) # d+6  d d d

    def forward(self, X_left, X_right, Feature_left, Feature_right):
        '''
        Input:
            X_left: Position of Left Child Node  n*6
            X_right: Position of Right Child Node  n*6
            Feature_left: Feature of Left Child Node  n*d
            Feature_right: Feature of Right Child Node  n*d
        Output:
            out_feature: Feature of Father Node n*d
        '''
        input_left = torch.cat((X_left, Feature_left),1)  # (n, 6+d)
        input_right = torch.cat((X_right, Feature_right),1)  # (n, 6+d)
        out_feature = self.W(input_left) + self.W(input_right) # (n, d)
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
            P_father: Position of Father Node  n*6
        Output:
            left_featrue: Feature of Left Child Node  n*d
            left_P: Position of Left Child Node  n*6
            left_isleaf: Left Child Node is leaf node or not (True or False)  n*1
            right_featrue: Feature of Right Child Node  n*d
            right_P: Position of Right Child Node  n*6
            right_isleaf: Right Child Node is leaf node or not (True or False)  n*1
        ''' 
        input_father = torch.cat((Feature_father, P_father),1)  # (n, d+6)
        out_father = self.M(input_father)  # (n, (d+6+1)*2)
        
        left_featrue = out_father[:,:self.n_feature]  # (n, d)
        left_P_xy = self.tanh(out_father[:,self.n_feature:(self.n_feature + 2)])  # (n, 2)
        left_P_s_wh = self.sigmoid(out_father[:,(self.n_feature + 2):(self.n_feature + 5)])  # (n, 3)
        left_P_a = self.tanh(out_father[:,(self.n_feature + 5):(self.n_feature + 6)])  # (n, 1)
        left_P = torch.cat((left_P_xy, left_P_s_wh, left_P_a), 1)  # (n, 6)
        left_isleaf = self.sigmoid(out_father[:,(self.n_feature + 6):(self.n_feature + 7)])  # (n, 1)
        
        right_featrue = out_father[:,(self.n_feature + 7):(self.n_feature * 2 + 7)]  # (n, d)
        right_P_xy = self.tanh(out_father[:,(self.n_feature * 2 + 7):(self.n_feature * 2 + 9)])  # (n, 2)
        right_P_s_wh = self.sigmoid(out_father[:,(self.n_feature * 2 + 9):(self.n_feature * 2 + 12)])  # (n, 3)
        right_P_a = self.tanh(out_father[:,(self.n_feature * 2 + 12):(self.n_feature * 2 + 13)])  # (n, 1)
        right_P = torch.cat((right_P_xy, right_P_s_wh, right_P_a), 1)  # (n, 6)
        right_isleaf = self.sigmoid(out_father[:,(self.n_feature * 2 + 13):])  # (n, 1)
        
        return left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf

class AE(SaveableModule):
    def __init__(self, device, m=25, n=25, weight=1, model_folder='log', save_name='ae', n_feature = 16, encode_in_channel = 6, decode_in_channel = 19):
        super(AE, self).__init__()

        self.n_feature = n_feature
        self.encoder = Encoder(n_feature, (n_feature+6))
        self.decoder = Decoder(n_feature, (n_feature+6))
        self.weight = weight
        self.MODELS_DIR = model_folder
        self.DEFAULT_SAVED_NAME =  save_name
        self.device = device
        self.m = m
        self.n = n

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
        X_r = X.clone()
        X_ab_xy = X.clone()
        X_ab_xy_r = X.clone()
        Feature = Feature_New.clone()
        Loss_P = 0.0
        Loss_Leaf = 0.0
        num = 0
        for i in range(num_I):
            I = I_list[num_I-1-i].squeeze(0)  # (ni, 3)

            p_left = X[I[:,0]]  # (ni, 6)
            p_right = X[I[:,1]]  # (ni, 6)
            leaf_left = Node_is_leaf[I[:,0]]  # (ni, 1)
            leaf_right = Node_is_leaf[I[:,1]]  # (ni, 1)

            p_father = X[I[:,2]]  # (ni, 6)
            f_father = Feature[I[:,2]]  # (ni, d)
            
            # (n, d), (n, 6), (n, 1), (n, d), (n, 6), (n, 1)
            left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf = self.decoder(f_father, p_father)
                    
            # update decoded Feature   
            X_r[I[:,0]] = left_P
            X_r[I[:,1]] = right_P
            Feature[I[:,0]] = left_featrue
            Feature[I[:,1]] = right_featrue

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
            left_xy_new = left_P[:,:2] * X_ab_xy_r[I[:,2],2:3] + X_ab_xy_r[I[:,2],:2]
            left_wh_new = left_P[:,3:5] * X_ab_xy_r[I[:,2],3:5]
            X_ab_xy_r[I[:,0],:2] = left_xy_new
            X_ab_xy_r[I[:,0],2] = left_P[:,2]
            X_ab_xy_r[I[:,0],3:5] = left_wh_new
            X_ab_xy_r[I[:,0],5] = left_P[:,5]

            right_xy_new = right_P[:,:2] * X_ab_xy_r[I[:,2],2:3] + X_ab_xy_r[I[:,2],:2]
            right_wh_new = right_P[:,3:5] * X_ab_xy_r[I[:,2],3:5]
            X_ab_xy_r[I[:,1],:2] = right_xy_new
            X_ab_xy_r[I[:,1],2] = right_P[:,2]
            X_ab_xy_r[I[:,1],3:5] = right_wh_new
            X_ab_xy_r[I[:,1],5] = right_P[:,5]

        return X_r, X_ab_xy, X_ab_xy_r, Feature, Loss_P, Loss_Leaf, num_I
        
    def forward(self, X, Feature, I_list, Node_is_leaf):
        '''
        Input:
            X: Positions of ALL Tree Nodes  B*n*6
            Feature: Features of ALL Tree Nodes  B*n*d
            I_list: The Index Matrix of merge Nodes  nlevel*B*ni*3
            Node_is_leaf: Leaf marks of ALL Tree Nodes  B*n*1
        Output:
            Loss: Sum Loss of ALL Father Nodes
            num: Number of Father Nodes
        ''' 
        X = X.squeeze(0)  # (n, 6)
        Feature = Feature.squeeze(0)  # (n, d)
        Node_is_leaf = Node_is_leaf.squeeze(0)  # (n, 1)
       
        Feature_New = self.encode(X, Feature, I_list)  # (n, d)
        # (n, 6), (n, 6), (n, 6), (n, d), (1), (1)
        X_r, X_ab_xy, X_ab_xy_r, Feature_r, Loss_P, Loss_Leaf, Num = self.decode(X, Node_is_leaf, Feature_New, I_list)
        # print(X_ab_xy, X_ab_xy_r)
        
        # index = torch.arange(len(Node_is_leaf)).view(-1,1)
        # idx = index[Node_is_leaf==1]
        # X_leaf = X_ab_xy[idx]
        # X_r_leaf = X_ab_xy_r[idx]
        # Loss_ab = self.cal_distance_ab(X_leaf, X_r_leaf)

        Loss_ab = self.cal_distance_ab(X_ab_xy, X_ab_xy_r, self.m, self.n)
        Loss_P = Loss_P / Num
        Loss_Leaf = Loss_Leaf / Num
        Loss = Loss_ab + Loss_P + Loss_Leaf
        Loss = Loss.requires_grad_()
        return Loss, Loss_ab, Loss_P, Loss_Leaf
    
    def rotate_edge_point(self, p, sin, cos, center):
        x_ = (p[:,:,0]-center[:,0:1])*cos-(p[:,:,1]-center[:,1:2])*sin+center[:,0:1]
        y_ = (p[:,:,0]-center[:,0:1])*sin+(p[:,:,1]-center[:,1:2])*cos+center[:,1:2]
        grid_r = torch.stack((x_, y_), 2)
        return grid_r

    def get_edge_point(self, P, F, m, n):
        ld = torch.cat((P[:,0:1]-F[:,0:1]/2, P[:,1:2]-F[:,1:2]/2),dim=1)
        rd = torch.cat((P[:,0:1]+F[:,0:1]/2, P[:,1:2]-F[:,1:2]/2),dim=1)
        ru = torch.cat((P[:,0:1]+F[:,0:1]/2, P[:,1:2]+F[:,1:2]/2),dim=1)
        lu = torch.cat((P[:,0:1]-F[:,0:1]/2, P[:,1:2]+F[:,1:2]/2),dim=1)
        x = torch.stack([ld[:,0] + (rd[:,0] - ld[:,0]) / (m-1) * i for i in range(m)], 1) 
        y = torch.stack([rd[:,1] + (ru[:,1] - rd[:,1]) / (n-1) * i for i in range(n)], 1) 
        
        idx_m = torch.arange(m-1, -1, -1)
        idx_n = torch.arange(n-1, -1, -1)

        grid_down = torch.stack((x, ld[:,1:2].expand(-1,m)), 2)
        grid_right = torch.stack((rd[:,0:1].expand(-1,n), y), 2)
        grid_up = torch.stack((x[:,idx_m], ru[:,1:2].expand(-1,m)), 2)
        grid_left = torch.stack((lu[:,0:1].expand(-1,n), y[:,idx_n]), 2) 
        
        grid = torch.cat((grid_down, grid_right,grid_up,grid_left),dim=1) 
        sinO = F[:,2:3] 
        cosO = torch.cos(torch.asin(F[:,2:3]))
        grid_r = self.rotate_edge_point(grid, sinO, cosO, P)
        return grid_r

    def cal_distance_ab(self, q, p, m, n):
        '''
        Input:
            p, q: position of a node  n*6
        Output:
            dis: The distance of two nodes
        ''' 
        # print(p)
        p_box = self.get_edge_point(p[:,:2], p[:,3:], m, n)
        q_box = self.get_edge_point(q[:,:2], q[:,3:], m, n)
        # print(p_box.shape, q_box.shape)
        dis = ChamfersDistance(p_box, q_box)
        # dis = F.mse_loss(p_box, q_box, reduction='mean')
        # dis = F.l1_loss(p_box, q_box, reduction='mean')
        # dis = F.smooth_l1_loss(p_box, q_box, reduction='mean')

        # print(p_box.shape)
        # dis_xy = F.mse_loss(p[:,:2], q[:,:2], reduction='mean')
        # dis_s_wha =  F.l1_loss(p[:,3:], q[:,3:], reduction='mean')
        # dis = dis_xy + dis_s_wha
        return dis
    

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

