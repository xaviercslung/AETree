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
        pen_state = np.zeros((node_xys.shape[0], 3))
        pen_state[:,0] = 1
        for i in range(5):
            pen_state[i*127:i*127+64]=np.array([0,1,0])
        node_xys = np.hstack((node_xys, pen_state))
        I_list = self.I_List[idx]
        I_list = [t.astype('int64') for t in I_list ]
        node_fea = torch.zeros(node_xys.shape[0], self.n_feature)
        return node_xys, I_list, node_fea
    
    def __len__(self):
        return len(self.node_list)

class Encoder(nn.Module):
    def __init__(self, in_channel = 8, n_feature = 128):
        super(Encoder, self).__init__()
        self.n_feature = n_feature
        out_channel = n_feature
        self.rnn = nn.LSTMCell(in_channel, n_feature//2)
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
    def __init__(self, n_feature = 128, M=20):
        super(Decoder, self).__init__()
        self.n_feature = n_feature
        self.M = M
        out_channel = (n_feature + 8) * 2
        # self.M = get_MLP_layers((in_channel, n_feature, n_feature*2, out_channel))
        self.fc_h = nn.Linear(n_feature, 2 * n_feature)
        self.rnn = nn.LSTMCell(n_feature + 8, n_feature)
        self.fc = nn.Linear(n_feature // 2, 13 * M + 3)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def get_parameter(self, y):
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, mu_a, mu_b, sigma_a, sigma_b, rho_ab, mu_slope, sigma_slope = torch.chunk(y[:,:-3], 13, 1)
        params_pen = y[:,-3:]
        pi = F.softmax(pi)
        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        sigma_a = torch.exp(sigma_a)
        sigma_b = torch.exp(sigma_b)
        sigma_slope = torch.exp(sigma_slope)
        rho_xy = torch.tanh(rho_xy)
        rho_ab = torch.tanh(rho_ab)
        q = F.softmax(params_pen)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, mu_a, mu_b, sigma_a, sigma_b, rho_ab, mu_slope, sigma_slope, q

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
        # Feature_father FC ???
        # print(Feature_father.shape)
        z_father = self.tanh(self.fc_h(Feature_father))
        h_father, c_father = torch.chunk(z_father, 2, 1)
        # print(z_father.shape, h_father.shape, c_father.shape)
        input_father = torch.cat((P_father, Feature_father), 1)  # (n, d+3)
        # print(input_father.shape)
        h_father_o, c_father_o = self.rnn(input_father, (h_father, c_father))  # (n, (d+3+1)*2)
        # print(h_father_o.shape, c_father_o.shape)

        h_left, h_right = torch.chunk(h_father_o, 2, 1)
        c_left, c_right = torch.chunk(c_father_o, 2, 1)

        y_left = self.fc(h_left)
        parameter_left = self.get_parameter(y_left)
        y_right = self.fc(h_right)
        parameter_right = self.get_parameter(y_right)

        return parameter_left, h_left, c_left, parameter_right, h_right, c_right

class AE(SaveableModule):
    def __init__(self, device, M=20, weight=1, model_folder='log', save_name='ae', n_feature = 16, encode_in_channel = 6, decode_in_channel = 19):
        super(AE, self).__init__()

        self.n_feature = n_feature
        self.M = M
        self.encoder = Encoder(8, n_feature)
        self.decoder = Decoder(n_feature, M)
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
    
    def decode(self, X, Feature_New, I_list):
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
        X_P = X.clone()
        Feature = Feature_New.clone()
        Loss = 0.0
        for i in range(num_I):
            I = I_list[num_I-1-i].squeeze(0)  # (n, ni)

            p_left = X[I[:,0]]  # (ni, 8)
            p_right = X[I[:,1]]  # (ni, 8)

            p_father = X[I[:,2]]  # (ni, 8)
            f_father = Feature[I[:,2]]  # (ni, d)
            
            # (14, n, 20 or 3), (n, 64), (n, 64), (14, n, 20 or 3), (n, 64), (n, 64)
            parameter_left, h_left, c_left, parameter_right, h_right, c_right = self.decoder(f_father, p_father)

            # expand left GT and right GT 
            p_left_repeat = self.make_target(p_left)
            p_right_repeat = self.make_target(p_right)
            loss, l_r_index = self.get_loss(p_left_repeat, parameter_left, p_right_repeat, parameter_right)

            #sample
            sample_left = p_left.clone()
            sample_right = p_right.clone()

            for n in range(len(p_father)):
                parameter = [parameter_left[i][n] for i in range(len(parameter_left))]
                sample_left[n] = self.sample_from_parameter(parameter)

                parameter = [parameter_right[i][n] for i in range(len(parameter_right))]
                sample_right[n] = self.sample_from_parameter(parameter)

            feature_left = torch.cat((h_left, c_left),1)
            feature_right = torch.cat((h_right, c_right),1)

            # assume the decoded leaf node is leaf node
            out_p_left = sample_left.clone()
            out_p_right = sample_right.clone()
            out_f_left = feature_left.clone()
            out_f_right = feature_right.clone()

            # swap the decoded leaf node and decoded right node according to loss 
            swap_index = torch.arange(len(l_r_index))[l_r_index==1]
            out_p_left[swap_index] = sample_right[swap_index]
            out_f_left[swap_index] = feature_right[swap_index]
            out_p_right[swap_index] = sample_left[swap_index]
            out_f_right[swap_index] = feature_left[swap_index]
            
            # update decoded Feature        
            Feature[I[:,0]] = out_f_left
            Feature[I[:,1]] = out_f_right
            X_P[I[:,0]] = out_p_left
            X_P[I[:,1]] = out_p_right

            Loss += loss
        return X_P, Feature, Loss/num_I
        
    def forward(self, X, Feature, I_list):
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
        Feature_New = self.encode(X, Feature, I_list)  # (n, d)
        # (n, 6), (n, 6), (n, d), (1), (1)
        X_P, Feature, Loss = self.decode(X, Feature_New, I_list)
        Loss = Loss.requires_grad_()
        return Loss
    
    def make_target(self, batch):

        dx = torch.stack([batch[:,0]]*self.M, 1)
        dy = torch.stack([batch[:,1]]*self.M, 1)
        da = torch.stack([batch[:,2]]*self.M, 1)
        db = torch.stack([batch[:,3]]*self.M, 1)
        dslope = torch.stack([batch.data[:,4]] * self.M, 1)
        p1 = batch.data[:,5]
        p2 = batch.data[:,6]
        p3 = batch.data[:,7]
        p = torch.stack([p1, p2, p3],1)
        return dx, dy, da, db, dslope, p

    def bivariate_normal_pdf(self, point, parameter):

        dx, dy, da, db, dslope, p = [point[i] for i in range(len(point))]
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, mu_a, mu_b, sigma_a, sigma_b, rho_ab, mu_slope, sigma_slope, q= [parameter[i] for i in range(len(parameter))]

        z_x = ((dx-mu_x)/sigma_x)**2
        z_y = ((dy-mu_y)/sigma_y)**2
        z_xy = (dx-mu_x)*(dy-mu_y)/(sigma_x*sigma_y)
        Z_xy = z_x + z_y -2*rho_xy*z_xy
        exp_xy = torch.exp(-Z_xy/(2*(1-rho_xy**2)))
        norm_xy = 2*np.pi*sigma_x*sigma_y*torch.sqrt(1-rho_xy**2)
        result_xy=exp_xy/norm_xy
        
        z_a = ((da-mu_a)/sigma_a)**2
        z_b = ((db-mu_b)/sigma_b)**2
        z_ab = (da-mu_a)*(db-mu_b)/(sigma_a*sigma_b)
        Z_ab = z_a + z_b -2*rho_ab*z_ab
        exp_ab = torch.exp(-Z_ab/(2*(1-rho_ab**2)))
        pi = torch.asin(torch.tensor(1.))*2
        norm_ab = 2*pi*sigma_a*sigma_b*torch.sqrt(1-rho_ab**2)
        result_ab=exp_ab/norm_ab

        z_slope=((dslope-mu_slope)/sigma_slope)**2
        exp_slope=torch.exp(-z_slope/2)
        norm_slope=torch.sqrt(2*2*torch.acos(torch.zeros(1)).cuda())*sigma_slope
        result_slope=exp_slope/norm_slope

        return result_xy, result_ab, result_slope

    def reconstruction_loss(self, point, parameter):
        
        pdf_xy, pdf_ab, pdf_slope = self.bivariate_normal_pdf(point, parameter)
        p = point[-1]
        pi = parameter[0]
        q = parameter[-1]

        LS_xy = -torch.log(1e-5+torch.sum(pi * pdf_xy, 1))
        LS_ab = -torch.log(1e-5+torch.sum(pi * pdf_ab, 1))
        LS_slope = -torch.log(1e-5+torch.sum(pi * pdf_slope, 1))
        LS=LS_xy+LS_ab+LS_slope
        LP = -torch.sum(p*torch.log(q), 1)
        return LS + LP

    def get_loss(self, p_left_repeat, parameter_left, p_right_repeat, parameter_right):

        l_l_r_r = self.reconstruction_loss(p_left_repeat, parameter_left) + self.reconstruction_loss(p_right_repeat, parameter_right)
        l_r_l_r = self.reconstruction_loss(p_left_repeat, parameter_right) + self.reconstruction_loss(p_right_repeat, parameter_left)

        loss, index = torch.min(torch.stack((l_l_r_r, l_r_l_r),0),0)
        Nmax = len(loss)
        loss = torch.sum(loss)/float(Nmax)

        return loss, index

    def sample_from_parameter(self, parameter, temperature=0.8):
        
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, mu_a, mu_b, sigma_a, sigma_b, rho_ab, mu_slope, sigma_slope, q = parameter

        def adjust_temp(pi_pdf):
            pi_pdf = torch.log(pi_pdf)/temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = torch.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        def sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy, temperature, greedy=False):
            # inputs must be floats
            if greedy:
                return mu_x,mu_y
            mean = torch.tensor([mu_x,mu_y])
            sigma_x *= torch.sqrt(torch.tensor(temperature))
            sigma_y *= torch.sqrt(torch.tensor(temperature))
            cov = torch.tensor([[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y], [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]])
            m = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
            x =  m.sample()
            return x[0], x[1]

        def sample_normal(mu_x,sigma_x,temperature):
            sigma_x *=torch.sqrt(torch.tensor(temperature))
            x= torch.normal(mu_x, sigma_x)
            return x 

        # get mixture indice:
        pi = adjust_temp(pi)
        pi_idx = torch.multinomial(pi, 1)
        # get pen state:
        q = adjust_temp(q)
        q_idx = torch.multinomial(q, 1)

        # get mixture params:
        mu_x = mu_x[pi_idx]
        mu_y = mu_y[pi_idx]
        sigma_x = sigma_x[pi_idx]
        sigma_y = sigma_y[pi_idx]
        rho_xy = rho_xy[pi_idx]

        mu_a = mu_a[pi_idx]
        mu_b = mu_b[pi_idx]
        sigma_a = sigma_a[pi_idx]
        sigma_b = sigma_b[pi_idx]
        rho_ab = rho_ab[pi_idx]

        mu_slope = mu_slope[pi_idx]
        sigma_slope = sigma_slope[pi_idx]
        
        x,y = sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy,temperature,greedy=False)
        a,b = sample_bivariate_normal(mu_a,mu_b,sigma_a,sigma_b,rho_ab,temperature,greedy=False)
        slope=sample_normal(mu_slope,sigma_slope,temperature)
        
        next_state = torch.zeros(8)
        next_state[0] = x
        next_state[1] = y
        next_state[2] = a
        next_state[3] = b
        next_state[4] = slope
        next_state[q_idx+5] = 1

        next_state = next_state.to(self.device)
        return next_state
    
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
        num = 0
        with torch.no_grad():
            for i,(node_xys, I_list, node_fea) in enumerate(loader, 0):
                node_xys = node_xys.to(device)
                node_xys = node_xys.float()
                I_list = [t.to(device) for t in I_list]
                node_fea = node_fea.to(device)
                node_fea = node_fea.float()
                loss = self.forward(node_xys,node_fea,I_list)
                total += loss
                num += 1
        return total/num

if __name__  == '__main__':

    Dataset = TreeData(root_dir)
    train_loader = torch.utils.data.DataLoader(Dataset, batch_size=1)
    node_xys, I_list, node_fea, node_is_leaf = next(iter(train_loader))
    model = AE()
    out = model(node_xys, node_fea, I_list,  node_is_leaf)

