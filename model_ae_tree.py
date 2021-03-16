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
        layers.append(get_and_init_FC_layer(dims[i - 1], dims[i]))
        if i == len(dims) - 1 and not doLastRelu:
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
        torch.save(self.state_dict(), os.path.join(self.MODELS_DIR, name + MODELS_EXT))

    def load_from_drive(model, name=None, model_dir=None, **kwargs):
        name = name if name is not None else model.DEFAULT_SAVED_NAME
        loaded = model(**kwargs)
        loaded.load_state_dict(torch.load(os.path.join(model_dir, name + MODELS_EXT)))
        loaded.eval()
        return loaded


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # print(sinusoid_table.shape)
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        # CPU tensor
        sinusoid_table = torch.FloatTensor(sinusoid_table)
        sinusoid_table = sinusoid_table.to(self.device)
        return sinusoid_table

    def forward(self, x, n):
        # print(x.shape, self.pos_table.shape,self.pos_table[n, :].shape)
        return x + self.pos_table[n, :]


class TreeData(Dataset):
    # the data folder should be data, change dir to ./data/Tree_2000_64_batch5.pickle
    def __init__(self, data_folder='./Tree_2000_64_batch5.pickle', train=True, split=0.8, n_feature=16):
        # load pickle file from dir:'./Tree_2000_64_batch5.pickle'
        self.data_folder = data_folder
        # set number of features
        self.n_feature = n_feature
        # read file in binary format
        node_list, I_List = pickle.load(open(data_folder, "rb"))  # whats inside node_list and I_List?
        num = len(node_list)
        # if train == True, where set it within init parameters
        if train:
            # take 80% of the data in node_list and I_List
            self.node_list = node_list[:int(num * split)]
            self.I_List = I_List[:int(num * split)]
        else:
            # take 20% of the data in node_list and I_List
            self.node_list = node_list[int(num * split): num]
            self.I_List = I_List[int(num * split): num]

    def __getitem__(self, idx):
        # torch.is_tensor is simply checking if idx is a PyTorch tensor.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # picking out index elements from node_list and I_List
        node_xys = self.node_list[idx]
        I_list = self.I_List[idx]
        # change every element inside I_list to int
        I_list = [t.astype('int64') for t in I_list]
        # creat an array of zeros. rows = node_xys.shape[0], columns = n_feature(which is 16 in this case)
        node_fea = torch.zeros(node_xys.shape[0], self.n_feature)

        # node_fea = np.hstack((self.node_list[idx][:,:2],self.node_list[idx][:,3:6]))
        # node_fea = self.node_list[idx][:,3:6]

        # create a CPU tensor, which occupies CPU memory. read more@ https://pytorch.org/docs/stable/tensors.html

        # view() --> change dimension of a tensor. read more@ https://pytorch.org/docs/stable/tensor_view.html
        node_is_leaf = torch.FloatTensor([64 * [1] + 63 * [0]] * 5).view(-1, 1)
        return node_xys, I_list, node_fea, node_is_leaf

    # helper method
    def __len__(self):
        return len(self.node_list)


class Encoder(nn.Module):
    def __init__(self, n_feature=16, in_channel=16 + 3):
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
        input_left = torch.cat((X_left, Feature_left), 1)  # (n, 6+d)
        input_right = torch.cat((X_right, Feature_right), 1)  # (n, 6+d)
        out_feature = self.W(input_left) + self.W(input_right)  # (n, d)

        return out_feature


class Decoder(nn.Module):
    def __init__(self, n_feature=16, in_channel=19):
        super(Decoder, self).__init__()
        self.n_feature = n_feature
        out_channel = (n_feature + 5 + 1) * 2
        self.M = get_MLP_layers((in_channel, n_feature, n_feature * 2, out_channel))
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
        input_father = torch.cat((Feature_father, P_father), 1)  # (n, d+3)
        out_father = self.M(input_father)  # (n, (d+5+1)*2)

        left_featrue = out_father[:, :self.n_feature]  # (n, d)
        left_P_xy = self.tanh(out_father[:, self.n_feature:(self.n_feature + 2)])  # (n, 2)
        left_P_wh = self.sigmoid(out_father[:, (self.n_feature + 2):(self.n_feature + 4)])  # (n, 2)
        left_P_a = out_father[:, (self.n_feature + 4):(self.n_feature + 5)]  # (n, 1)
        left_P = torch.cat((left_P_xy, left_P_wh, left_P_a), 1)  # (n, 5)
        left_isleaf = self.sigmoid(out_father[:, (self.n_feature + 5):(self.n_feature + 6)])  # (n, 1)

        right_featrue = out_father[:, (self.n_feature + 6):(self.n_feature * 2 + 6)]  # (n, d)
        right_P_xy = self.tanh(out_father[:, (self.n_feature * 2 + 6):(self.n_feature * 2 + 8)])  # (n, 2)
        right_P_s_wh = self.sigmoid(out_father[:, (self.n_feature * 2 + 8):(self.n_feature * 2 + 10)])  # (n, 2)
        right_P_a = out_father[:, (self.n_feature * 2 + 10):(self.n_feature * 2 + 11)]  # (n, 1)
        right_P = torch.cat((right_P_xy, right_P_s_wh, right_P_a), 1)  # (n, 5)
        right_isleaf = self.sigmoid(out_father[:, (self.n_feature * 2 + 11):])  # (n, 1)

        return left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf


class AE(SaveableModule):
    def __init__(self, device, weight_type=0, weight=1, model_folder='log', save_name='ae', n_feature=16,
                 encode_in_channel=6, decode_in_channel=19):

        # same as super().__init__()
        super(AE, self).__init__()

        self.n_feature = n_feature
        self.encoder = Encoder(n_feature, n_feature + 5)
        self.decoder = Decoder(n_feature, n_feature + 5)
        # self.G = get_MLP_layers((2, n_feature//4, n_feature//2, n_feature))
        self.weight = weight
        self.weight_type = weight_type
        self.MODELS_DIR = model_folder
        self.DEFAULT_SAVED_NAME = save_name
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
        # torch.clone()
        Feature_New = Feature.clone()
        num_I = len(I_list)

        position_enc = PositionalEncoding(self.n_feature, n_position=num_I, device=self.device)

        for i in range(num_I):
            I = I_list[i].squeeze(0)  # (ni, 3)
            left_p = X[I[:, 0]]  # (ni, 6)
            right_p = X[I[:, 1]]  # (ni, 6)
            left_f = Feature_New[I[:, 0]]  # (ni, d)
            right_f = Feature_New[I[:, 1]]  # (ni, d)

            left_f = position_enc(left_f, i)
            right_f = position_enc(right_f, i)

            out = self.encoder(left_p, right_p, left_f, right_f)  # (ni, d) 
            Feature_New[I[:, 2]] = out

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

        position_enc = PositionalEncoding(self.n_feature, n_position=num_I, device=self.device)

        for i in range(num_I):
            I = I_list[num_I - 1 - i].squeeze(0)  # (n, ni)

            p_left = X_ab_xy[I[:, 0]]  # (ni, 6)
            p_right = X_ab_xy[I[:, 1]]  # (ni, 6)
            leaf_left = Node_is_leaf[I[:, 0]]  # (ni, 1)
            leaf_right = Node_is_leaf[I[:, 1]]  # (ni, 1)

            p_father = X_ab_xy[I[:, 2]]  # (ni, 6)
            f_father = Feature[I[:, 2]]  # (ni, d)
            f_father = position_enc(f_father, num_I - i - 1)
            # (n, d), (n, 6), (n, 1), (n, d), (n, 6), (n, 1)
            left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf = self.decoder(f_father, p_father)

            weight = 1

            if (self.weight_type == 1):
                weight = (2 ** (num_I - i - 1))

            if (self.weight_type == 2):
                weight = (2 * (num_I - i))

            left_re_loss = self.cal_distance_re(p_left, left_P)
            right_re_loss = self.cal_distance_re(p_right, right_P)
            # print(self.weight_type, weight)
            Loss_P = Loss_P + (left_re_loss + right_re_loss) * weight

            left_check.append(left_re_loss)
            right_check.append(right_re_loss)

            # update decoded Feature
            X_r[I[:, 0]] = left_P
            X_r[I[:, 1]] = right_P
            Feature[I[:, 0]] = left_featrue
            Feature[I[:, 1]] = right_featrue

            # get GroundTruth absolute xy
            left_xy_new = X_ab_xy[I[:, 0], :2] * X_ab_xy[I[:, 2], 2:4] + X_ab_xy[I[:, 2], :2]
            X_ab_xy[I[:, 0], :2] = left_xy_new
            left_wh_new = X_ab_xy[I[:, 0], 2:4] * X_ab_xy[I[:, 2], 2:4]
            X_ab_xy[I[:, 0], 2:4] = left_wh_new
            left_a_new = X_ab_xy[I[:, 0], 4] + X_ab_xy[I[:, 2], 4]
            X_ab_xy[I[:, 0], 4] = left_a_new

            right_xy_new = X_ab_xy[I[:, 1], :2] * X_ab_xy[I[:, 2], 2:4] + X_ab_xy[I[:, 2], :2]
            X_ab_xy[I[:, 1], :2] = right_xy_new
            right_wh_new = X_ab_xy[I[:, 1], 2:4] * X_ab_xy[I[:, 2], 2:4]
            X_ab_xy[I[:, 1], 2:4] = right_wh_new
            right_a_new = X_ab_xy[I[:, 1], 4] + X_ab_xy[I[:, 2], 4]
            X_ab_xy[I[:, 1], 4] = right_a_new

            # get Reconstruction absolute xy
            left_xy_new = left_P[:, :2] * X_ab_xy_r[I[:, 2], 2:4] + X_ab_xy_r[I[:, 2], :2]
            left_wh_new = left_P[:, 2:4] * X_ab_xy_r[I[:, 2], 2:4]
            left_a_new = left_P[:, 4] + X_ab_xy_r[I[:, 2], 4]
            X_ab_xy_r[I[:, 0], :2] = left_xy_new
            X_ab_xy_r[I[:, 0], 2:4] = left_wh_new
            X_ab_xy_r[I[:, 0], 4] = left_a_new

            right_xy_new = right_P[:, :2] * X_ab_xy_r[I[:, 2], 2:4] + X_ab_xy_r[I[:, 2], :2]
            right_wh_new = right_P[:, 2:4] * X_ab_xy_r[I[:, 2], 2:4]
            right_a_new = right_P[:, 4] + X_ab_xy_r[I[:, 2], 4]
            X_ab_xy_r[I[:, 1], :2] = right_xy_new
            X_ab_xy_r[I[:, 1], 2:4] = right_wh_new
            X_ab_xy_r[I[:, 1], 4] = right_a_new

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
        X_r, X_ab_xy, X_ab_xy_r, Feature_r, Loss_P, Loss_Leaf, Num, left_check, right_check = self.decode(X,
                                                                                                          Node_is_leaf,
                                                                                                          Feature_New,
                                                                                                          I_list)
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
        Loss_Leaf = Loss_Leaf / Num
        Loss = Loss_ab + Loss_P + Loss_Leaf
        # print(Loss, Loss_ab , Loss_P , Loss_Leaf)
        # Loss = Loss / Num
        Loss = Loss.requires_grad_()
        return Loss, Loss_ab, Loss_P, Loss_Leaf, left_check, right_check

    def cal_distance(self, p, q):
        '''
        Input:
            p, q: position of a node  n*3
        Output:
            dis: The distance of two nodes
        '''
        px, py, ps = p[:, 0], p[:, 1], p[:, 2]
        qx, qy, qs = q[:, 0], q[:, 1], q[:, 2]

        lx = torch.max(px + ps / 2, qx + qs / 2) - torch.min(px - ps / 2, qx - qs / 2)
        ly = torch.max(py + ps / 2, qy + qs / 2) - torch.min(py - ps / 2, qy - qs / 2)
        dis = torch.abs(ps * ps + ps * ps - 2 * lx * ly)
        return dis

    def cal_distance2(self, q, p, details=False):
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

        dis_xy = F.mse_loss(p[:, :2], q[:, :2], reduction='none')
        dis_xy = torch.sum(dis_xy, 1)
        dis_wha = F.l1_loss(p[:, 2:], q[:, 2:], reduction='none')
        dis_wha = torch.sum(dis_wha, 1)
        dis = dis_xy + dis_wha

        # dis_xywh =  F.mse_loss(p[:,:4], q[:,:4], reduction='none')
        # dis_xywh = torch.sum(dis_xywh, 1)
        # dis_a =  F.l1_loss(p[:,4:], q[:,4:], reduction='none')
        # dis_a = torch.sum(dis_a, 1)
        # dis = dis_xywh + dis_a
        return dis

    def rotate_xy(self, p, sin, cos, center):
        x_ = (p[:, 0:1] - center[:, 0:1]) * cos - (p[:, 1:2] - center[:, 1:2]) * sin + center[:, 0:1]
        y_ = (p[:, 0:1] - center[:, 0:1]) * sin + (p[:, 1:2] - center[:, 1:2]) * cos + center[:, 1:2]
        #     print(((p[:,0:1]-center[:,0:1])*cos).shape, cos.shape, x_.shape)
        return torch.cat((x_, y_), 1)

    def get_box(self, P, F):
        # print(P.shape, F.shape)
        ld = torch.cat((P[:, 0:1] - F[:, 0:1] / 2, P[:, 1:2] - F[:, 1:2] / 2), -1)
        rd = torch.cat((P[:, 0:1] + F[:, 0:1] / 2, P[:, 1:2] - F[:, 1:2] / 2), -1)
        ru = torch.cat((P[:, 0:1] + F[:, 0:1] / 2, P[:, 1:2] + F[:, 1:2] / 2), -1)
        lu = torch.cat((P[:, 0:1] - F[:, 0:1] / 2, P[:, 1:2] + F[:, 1:2] / 2), -1)
        # print((P[:,0:1]-F[:,0:1]/2).shape, rd.shape)
        # box = np.hstack((ld, rd, ru, lu)).reshape(len(P), -1, 2)
        sinO = torch.sin(F[:, 2:3])
        cosO = torch.cos(F[:, 2:3])

        ld_r = self.rotate_xy(ld, sinO, cosO, P)
        rd_r = self.rotate_xy(rd, sinO, cosO, P)
        ru_r = self.rotate_xy(ru, sinO, cosO, P)
        lu_r = self.rotate_xy(lu, sinO, cosO, P)
        if (len(P) > 0):
            box_r = torch.cat((ld_r, rd_r, ru_r, lu_r), 1)
            # print(box_r.shape)
        else:
            box_r = []
        return box_r

    def cal_distance_ab(self, q, p):
        '''
        Input:
            p, q: position of a node  n*2
        Output:
            dis: The distance of two nodes
        '''
        # print(p)
        p_box = self.get_box(p[:, :2], p[:, 2:])
        q_box = self.get_box(q[:, :2], q[:, 2:])
        # dis = F.mse_loss(p_box, q_box, reduction='mean')
        dis = F.l1_loss(p_box, q_box, reduction='mean')
        # dis = F.smooth_l1_loss(p_box, q_box, reduction='mean')
        # print(p_box.shape)
        # dis_xy = F.mse_loss(p[:,:2], q[:,:2], reduction='mean')
        # dis_s_wha =  F.l1_loss(p[:,3:], q[:,3:], reduction='mean')
        # dis = dis_xy + dis_s_wha
        return dis

    def cal_distance_re(self, q, p):
        '''
        Input:
            p, q: position of a node  n*3
        Output:
            dis: The distance of two nodes
        '''
        dis_xy = F.mse_loss(p[:, :2], q[:, :2], reduction='mean')
        dis_wh = F.mse_loss(p[:, 2:4], q[:, 2:4], reduction='mean')
        dis_a = F.l1_loss(p[:, 4:], q[:, 4:], reduction='mean')
        dis = dis_xy + dis_wh + dis_a
        # dis = dis_xy + dis_wh
        # dis =  F.l1_loss(p, q, reduction='mean')

        return dis

    def get_Binary_loss(self, left_is_leaf, left_is_leaf2, right_is_leaf, right_is_leaf2):

        left_loss = F.binary_cross_entropy(left_is_leaf2, left_is_leaf, reduction='none')
        right_loss = F.binary_cross_entropy(right_is_leaf2, right_is_leaf, reduction='none')
        return left_loss + right_loss

    def get_Position_loss(self, leaf_p, leaf_p2, right_p, right_p2):
        # l_l_r_r = self.cal_distance(leaf_p, leaf_p2) + self.cal_distance(right_p, right_p2)
        # l_r_l_r = self.cal_distance(leaf_p, right_p2) + self.cal_distance(right_p, leaf_p2)

        l_l_r_r = self.cal_distance2(leaf_p, leaf_p2) + self.cal_distance2(right_p, right_p2)
        l_r_l_r = self.cal_distance2(leaf_p, right_p2) + self.cal_distance2(right_p, leaf_p2)
        loss, index = torch.min(torch.stack((l_l_r_r, l_r_l_r), 0), 0)

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
        if (n == 0):
            return out

        left_featrue, left_P, left_isleaf, right_featrue, right_P, right_isleaf = self.decoder(F, P)
        out.append([left_P, right_P, P])

        left_isleaf = torch.round(left_isleaf)
        right_isleaf = torch.round(right_isleaf)

        if (not left_isleaf):
            self.inference(left_P, left_featrue, n - 1, out)
        if (not right_isleaf):
            self.inference(right_P, right_featrue, n - 1, out)
        else:
            return out

    def loss_on_loader(self, loader, device):
        # calculate loss on all data
        total = 0.0
        total_ab = 0.0
        total_p = 0.0
        total_leaf = 0.0
        num = 0

        train_loss_left_check = np.zeros((10, 1))
        train_loss_right_check = np.zeros((10, 1))

        with torch.no_grad():
            for i, (node_xys, I_list, node_fea, node_is_leaf) in enumerate(loader, 0):
                node_xys = node_xys.to(device)
                node_xys = node_xys.float()
                I_list = [t.to(device) for t in I_list]
                node_fea = node_fea.to(device)
                node_fea = node_fea.float()
                node_is_leaf = node_is_leaf.to(device)
                loss, loss_ab, loss_p, loss_leaf, left_check, right_check = self.forward(node_xys, node_fea, I_list,
                                                                                         node_is_leaf)

                for i, ITEM in enumerate(left_check):
                    train_loss_left_check[i] += ITEM.item()
                for i, ITEM in enumerate(right_check):
                    train_loss_right_check[i] += ITEM.item()

                total += loss
                total_ab += loss_ab
                total_p += loss_p
                total_leaf += loss_leaf
                num += 1
        return total / num, total_ab / num, total_p / num, total_leaf / num, train_loss_left_check / num, train_loss_right_check / num


if __name__ == '__main__':
    # what does root_dir do? Where is the data?
    # get data from pickle file and performs manipulation
    Dataset = TreeData(root_dir)

    train_loader = torch.utils.data.DataLoader(Dataset, batch_size=1)
    node_xys, I_list, node_fea, node_is_leaf = next(iter(train_loader))
    model = AE()
    out = model(node_xys, node_fea, I_list, node_is_leaf)
