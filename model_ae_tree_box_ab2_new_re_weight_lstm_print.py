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

class Encoder(nn.Module):
    def __init__(self, n_feature = 16, in_channel = 16+3):
        super(Encoder, self).__init__()
        self.n_feature = n_feature
        out_channel = n_feature
        self.rnn = nn.LSTMCell(in_channel, n_feature//2)
        # self.W = get_MLP_layers((in_channel, n_feature, n_feature, n_feature))
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
        h_l, c_l = torch.chunk(Feature_left, 2, 1)
        h_r, c_r = torch.chunk(Feature_right, 2, 1)
        # print(h_l.shape, c_l.shape)
        h_l_o, c_l_o = self.rnn(X_left, (h_l, c_l)) 
        h_r_o, c_r_o = self.rnn(X_right, (h_r, c_r)) 
        # print(h_l_o.shape, h_r_o.shape)
        h_o = h_l_o + h_r_o
        c_o = c_l_o + c_r_o
        # print(h_o.shape, c_o.shape)
        out_feature = torch.cat((h_o, c_o),1)
        # print(out_feature.shape)
        
        return out_feature
    
class Decoder(nn.Module):
    def __init__(self, n_feature = 16, in_channel = 19):
        super(Decoder, self).__init__()
        self.n_feature = n_feature
       
        self.fc_h = nn.Linear(n_feature, 2 * n_feature)
        self.rnn = nn.LSTMCell(n_feature + 5, n_feature)
        self.fc_l= nn.Linear(n_feature//2, 6)
        self.fc_r= nn.Linear(n_feature//2, 6)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

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

        z_father = self.relu(self.fc_h(Feature_father))
        h_father, c_father = torch.chunk(z_father, 2, 1)
        # print(z_father.shape, h_father.shape, c_father.shape)
        input_father = torch.cat((P_father, Feature_father), 1)  # (n, d+3)
        # print(input_father.shape)
        h_father_o, c_father_o = self.rnn(input_father, (h_father, c_father))  # (n, (d+3+1)*2)
        # print(h_father_o.shape, c_father_o.shape)

        h_left, h_right = torch.chunk(h_father_o, 2, 1)
        c_left, c_right = torch.chunk(c_father_o, 2, 1)

        y_left = self.fc_l(h_left)
        # y_left = self.drop(y_left)
        y_right = self.fc_r(h_right)
        # y_right = self.drop(y_right)

        left_P_xy = self.tanh(y_left[:,:2])  # (n, 2)
        left_P_wh = self.sigmoid(y_left[:,2:4])  # (n, 3)
        left_P_a = y_left[:,4:5]  # (n, 1)
        left_P = torch.cat((left_P_xy, left_P_wh, left_P_a), 1)  # (n, 6)
        left_isleaf = self.sigmoid(y_left[:,5:])

        right_P_xy = self.tanh(y_right[:,:2])  # (n, 2)
        right_P_wh = self.sigmoid(y_right[:,2:4])  # (n, 3)
        right_P_a = y_right[:,4:5]  # (n, 1)
        right_P = torch.cat((right_P_xy, right_P_wh, right_P_a), 1)  # (n, 6)
        right_isleaf = self.sigmoid(y_right[:,5:])

        left_featrue = torch.cat((h_left, c_left),1)
        right_featrue = torch.cat((h_right, c_right),1)
        
        return left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf
        

class AE(SaveableModule):
    def __init__(self, device, leaf_loss = False, weight_type=0, weight_leaf=1, model_folder='log', save_name='ae', n_feature = 16, encode_in_channel = 6, decode_in_channel = 19):
        super(AE, self).__init__()

        self.n_feature = n_feature
        self.encoder = Encoder(n_feature, 5)
        self.decoder = Decoder(n_feature, n_feature + 5)
        # self.G = get_MLP_layers((2, n_feature//4, n_feature//2, n_feature))
        self.weight_leaf = weight_leaf
        self.weight_type = weight_type
        self.leaf_loss = leaf_loss
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
        num_I = len(I_list)
        for i in range(num_I):
            I = I_list[i].squeeze(0)   # (ni, 3)
            left_p = X[I[:,0]]  # (ni, 6) 
            right_p = X[I[:,1]]  # (ni, 6) 
            left_f = Feature_New[I[:,0]]  # (ni, d) 
            right_f = Feature_New[I[:,1]]  # (ni, d)
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
        left_check = []
        right_check = []

        for i in range(num_I):
            I = I_list[num_I-1-i].squeeze(0)  # (n, ni)

            p_left = X_ab_xy[I[:,0]]  # (ni, 6)
            p_right = X_ab_xy[I[:,1]]  # (ni, 6)
            leaf_left = Node_is_leaf[I[:,0]]  # (ni, 1)
            leaf_right = Node_is_leaf[I[:,1]]  # (ni, 1)

            p_father = X_ab_xy[I[:,2]]  # (ni, 6)
            f_father = Feature[I[:,2]]  # (ni, d)
            # (n, d), (n, 6), (n, 1), (n, d), (n, 6), (n, 1)
            left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf = self.decoder(f_father, p_father)
            
            weight = 1

            if(self.weight_type==1):
                weight = (2**(num_I-i-1))
            
            if(self.weight_type==2):
                weight = (2*(num_I-i))


            left_re_loss = self.cal_distance_re(p_left, left_P)
            right_re_loss = self.cal_distance_re(p_right, right_P)
            # print(self.weight_type, weight)
            Loss_P = Loss_P + (left_re_loss + right_re_loss) * weight

            # calculate leaf loss
            if(self.leaf_loss):
                left_leaf_loss = self.cal_leaf_loss(leaf_left, left_isleaf)
                right_leaf_loss = self.cal_leaf_loss(leaf_right, right_isleaf)
                Loss_Leaf =  Loss_Leaf + (left_leaf_loss + right_leaf_loss) * weight
            
            left_check.append(left_re_loss)
            right_check.append(right_re_loss)

            # update decoded Feature
            X_r[I[:,0]] = left_P
            X_r[I[:,1]] = right_P        
            Feature[I[:,0]] = left_featrue
            Feature[I[:,1]] = right_featrue

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
            left_xy_new = left_P[:,:2] * X_ab_xy_r[I[:,2],2:4] + X_ab_xy_r[I[:,2],:2]
            left_wh_new = left_P[:,2:4] * X_ab_xy_r[I[:,2],2:4]
            left_a_new = left_P[:,4] + X_ab_xy_r[I[:,2],4]
            X_ab_xy_r[I[:,0],:2] = left_xy_new
            X_ab_xy_r[I[:,0],2:4] = left_wh_new
            X_ab_xy_r[I[:,0],4] = left_a_new

            right_xy_new = right_P[:,:2] * X_ab_xy_r[I[:,2],2:4] + X_ab_xy_r[I[:,2],:2]
            right_wh_new = right_P[:,2:4] * X_ab_xy_r[I[:,2],2:4]
            right_a_new = right_P[:,4] + X_ab_xy_r[I[:,2],4]
            X_ab_xy_r[I[:,1],:2] = right_xy_new
            X_ab_xy_r[I[:,1],2:4] = right_wh_new
            X_ab_xy_r[I[:,1],4] = right_a_new

        return X_r, X_ab_xy, X_ab_xy_r, Feature, Loss_P, Loss_Leaf, num_I, left_check, right_check
        
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
        X_r, X_ab_xy, X_ab_xy_r, Feature_r, Loss_P, Loss_Leaf, Num, left_check, right_check = self.decode(X, Node_is_leaf, Feature_New, I_list)
        # print(X_ab_xy, X_ab_xy_r)

        # index = torch.arange(len(Node_is_leaf)).view(-1,1)
        # idx = index[Node_is_leaf==1]
        # X_leaf = X_ab_xy[idx]
        # X_r_leaf = X_ab_xy_r[idx]
        # Loss_ab = self.cal_distance_ab(X_leaf, X_r_leaf)
        Loss_ab = 0.0
        # Loss_fea = self.cal_distance_fea(Feature_New, Feature_r)
        # Loss_fea = 0
        # Loss_P = self.cal_distance_re(X, X_r) * self.weight
        
        Loss_P = Loss_P / Num
        Loss_Leaf = Loss_Leaf / Num * self.weight_leaf
        Loss = Loss_ab + Loss_P + Loss_Leaf
        # print(Loss, Loss_ab , Loss_P , Loss_Leaf)
        # Loss = Loss / Num
        Loss = Loss.requires_grad_()
        return Loss, Loss_ab, Loss_P, Loss_Leaf, left_check, right_check
    
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
                node_fea = node_fea.to(device)
                node_fea = node_fea.float()
                node_is_leaf = node_is_leaf.to(device)
                loss, loss_ab, loss_p, loss_leaf, left_check, right_check = self.forward(node_xys,node_fea,I_list,node_is_leaf)
                
                for i,ITEM in enumerate(left_check):
                    train_loss_left_check[i] += ITEM.item()
                for i,ITEM in enumerate(right_check):
                    train_loss_right_check[i] += ITEM.item()

                total += loss
                total_ab += loss_ab
                total_p += loss_p
                total_leaf += loss_leaf
                num += 1
        return total/num, total_ab/num, total_p/num, total_leaf/num, train_loss_left_check/num, train_loss_right_check/num

if __name__  == '__main__':

    Dataset = TreeData(root_dir)
    train_loader = torch.utils.data.DataLoader(Dataset, batch_size=1)
    node_xys, I_list, node_fea, node_is_leaf = next(iter(train_loader))
    model = AE()
    out = model(node_xys, node_fea, I_list,  node_is_leaf)

