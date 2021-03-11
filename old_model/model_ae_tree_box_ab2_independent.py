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
        out_channel = (n_feature + 5 + 1) * 2
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
        out_father = self.M(input_father)  # (n, (d+5+1)*2)
        
        left_featrue = out_father[:,:self.n_feature]  # (n, d)
        left_P_xy = self.tanh(out_father[:,self.n_feature:(self.n_feature + 2)])  # (n, 2)
        left_P_wh = self.sigmoid(out_father[:,(self.n_feature + 2):(self.n_feature + 4)])  # (n, 2)
        left_P_a = out_father[:,(self.n_feature + 4):(self.n_feature + 5)]  # (n, 1)
        left_P = torch.cat((left_P_xy, left_P_wh, left_P_a), 1)  # (n, 5)
        left_isleaf = self.sigmoid(out_father[:,(self.n_feature + 5):(self.n_feature + 6)])  # (n, 1)
        
        right_featrue = out_father[:,(self.n_feature + 6):(self.n_feature * 2 + 6)]  # (n, d)
        right_P_xy = self.tanh(out_father[:,(self.n_feature * 2 + 6):(self.n_feature * 2 + 8)])  # (n, 2)
        right_P_s_wh = self.sigmoid(out_father[:,(self.n_feature * 2 + 8):(self.n_feature * 2 + 10)])  # (n, 2)
        right_P_a = out_father[:,(self.n_feature * 2 + 10):(self.n_feature * 2 + 11)]  # (n, 1)
        right_P = torch.cat((right_P_xy, right_P_s_wh, right_P_a), 1)  # (n, 5)
        right_isleaf = self.sigmoid(out_father[:,(self.n_feature * 2 + 11):])  # (n, 1)
        
        return left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf

class AE(SaveableModule):
    def __init__(self, device, leaf_loss = False, weight_type=0, weight_wh=1, weight=1, model_folder='log', save_name='ae', n_feature = 16, encode_in_channel = 6, decode_in_channel = 19):
        super(AE, self).__init__()

        self.n_feature = n_feature
        self.encoder = Encoder(n_feature, n_feature + 5)
        self.decoder = Decoder(n_feature, n_feature + 5)
        # self.G = get_MLP_layers((2, n_feature//4, n_feature//2, n_feature))
        self.weight = weight
        self.weight_wh = weight_wh
        self.weight_type = weight_type
        self.leaf_loss = leaf_loss
        self.MODELS_DIR = model_folder
        self.DEFAULT_SAVED_NAME =  save_name
        self.device = device

    def encode_decode(self, X, Node_is_leaf, Feature, I_list):
        '''
        Input:
            X: Positions of ALL Tree Nodes  n*3
            Feature: Features of ALL Tree Nodes  n*d
            I_list: The Index Matrix of merge Nodes  nlevel*B*ni*3
        Output:
            Feature: Encoded Features (updates on input Features)  B*n*d
        ''' 
        # Feature[:64] = self.G(X[:64,:2])
        
        num_I = len(I_list)  # nlevel
        X_r = X.clone()
        X_ab_xy = X.clone()
        X_ab_xy_r = X.clone()
        Feature_New = Feature.clone()

        Loss_P = 0.0
        Loss_Leaf = 0.0
        num = 0
        re_loss = []

        for i,item in enumerate(I_list):
            I = item.squeeze(0)  # (ni, 3)
            left_p = X[I[:,0]]  # (ni, 6) 
            right_p = X[I[:,1]]  # (ni, 6) 
            left_f = Feature_New[I[:,0]]  # (ni, d) 
            right_f = Feature_New[I[:,1]]  # (ni, d)
            leaf_left = Node_is_leaf[I[:,0]]  # (ni, 1)
            leaf_right = Node_is_leaf[I[:,1]]  # (ni, 1) 
            father_p = X[I[:,2]]
            # encode
            father_f = self.encoder(left_p, right_p, left_f, right_f)  # (ni, d) 
            # decode
            left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf = self.decoder(father_f, father_p)

            # calculate relative loss
            left_re_loss = self.cal_distance_re(left_p, left_P)
            right_re_loss = self.cal_distance_re(right_p, right_P)
            # set weight of relative loss
            weight = 1
            if(self.weight_type==1):
                weight = (2**(num_I-i-1))
            if(self.weight_type==2):
                weight = (2*(num_I-i))
            Loss_P = Loss_P + (left_re_loss + right_re_loss) * weight

            # calculate leaf loss
            if(self.leaf_loss):
                left_leaf_loss = self.cal_leaf_loss(leaf_left, left_isleaf)
                right_leaf_loss = self.cal_leaf_loss(leaf_right, right_isleaf)
                Loss_Leaf =  Loss_Leaf + (left_leaf_loss + right_leaf_loss)

            # update decoded Feature
            X_r[I[:,0]] = left_P
            X_r[I[:,1]] = right_P        
            Feature_New[I[:,0]] = left_featrue
            Feature_New[I[:,1]] = right_featrue
            Feature_New[I[:,2]] = father_f

        return X_r, X_ab_xy, X_ab_xy_r, Feature_New, Loss_P, Loss_Leaf, num_I
        
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
       
        X_r, X_ab_xy, X_ab_xy_r, Feature_r, Loss_P, Loss_Leaf, Num = self.encode_decode(X, Node_is_leaf, Feature, I_list)  # (n, d)
       
        Loss_ab = 0.0
        Loss_P = Loss_P / Num
        Loss_Leaf = Loss_Leaf / Num
        Loss = Loss_ab + Loss_P + Loss_Leaf
        # print(Loss, Loss_ab , Loss_P , Loss_Leaf)
        # Loss = Loss / Num
        Loss = Loss.requires_grad_()
        return Loss, Loss_ab, Loss_P, Loss_Leaf
    
    def cal_distance_re(self, q, p):
        '''
        Input:
            p, q: position of a node  n*3
        Output:
            dis: The distance of two nodes
        '''         
        dis_xy =  F.mse_loss(p[:,:2], q[:,:2], reduction='mean')
        dis_wh =  F.mse_loss(p[:,2:4], q[:,2:4], reduction='mean')
        dis_a =  F.l1_loss(p[:,4:], q[:,4:], reduction='mean')
        dis = dis_xy +  self.weight_wh * dis_wh + dis_a
        # dis = dis_xy + dis_wh
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

