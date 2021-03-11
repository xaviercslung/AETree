from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
MODELS_EXT = '.dms'
class STN5d(nn.Module):
    def __init__(self):
        super(STN5d, self).__init__()
        self.conv1 = torch.nn.Conv1d(5, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 25)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(5).flatten().astype(np.float32))).view(1,5*5).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 5, 5)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x




class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN5d()
        self.conv1 = torch.nn.Conv1d(5, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetEncoder(nn.Module):
    def __init__(self, k = 128, feature_transform=False):
        super(PointNetEncoder, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

class MLP_Decoder(nn.Module):
    def __init__(self, n_feature = 128, M=20):
        super(MLP_Decoder, self).__init__()
        self.n_feature = n_feature
        self.M = M
        # self.M = get_MLP_layers((in_channel, n_feature, n_feature*2, out_channel))
        self.fc_1 = nn.Linear(n_feature,  n_feature)
        self.fc_2 = nn.Linear(n_feature,  n_feature//2)
        self.fc_3 = nn.Linear(n_feature//2 , 13 * M)

    
    def get_parameter(self, y):
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, mu_a, mu_b, sigma_a, sigma_b, rho_ab, mu_slope, sigma_slope = torch.chunk(y, 13, 2)
        
        pi = F.softmax(pi)
        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        sigma_a = torch.exp(sigma_a)
        sigma_b = torch.exp(sigma_b)
        sigma_slope = torch.exp(sigma_slope)
        rho_xy = torch.tanh(rho_xy)
        rho_ab = torch.tanh(rho_ab)
        
        mu_x=torch.sigmoid(mu_x)
        mu_y=torch.sigmoid(mu_y)
        mu_a=torch.sigmoid(mu_a)
        mu_b=torch.sigmoid(mu_b)
        mu_slope=torch.sigmoid(mu_slope)
        
        
        return [pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, mu_a, mu_b, sigma_a, sigma_b, rho_ab, mu_slope, sigma_slope]

    def forward(self,feature):
        y=self.fc_1(feature)
        y=F.relu(y)
        y=self.fc_2(y)
        y=F.relu(y)
        y=self.fc_3(y)
        parameter = self.get_parameter(y)

        return parameter 

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



class pointnetbaseline(SaveableModule):
    def __init__(self, device, M=20, model_folder='log', save_name='baseline', n_feature = 128):
        super(pointnetbaseline, self).__init__()

        self.n_feature = n_feature
        self.M = M
        self.encoder = PointNetEncoder(k = n_feature )
        self.decoder = MLP_Decoder(n_feature, M)
       
        self.MODELS_DIR = model_folder
        self.DEFAULT_SAVED_NAME =  save_name
        self.device = device
    def make_target(self, batch):

        dx = torch.stack([batch[:,0,:]]*self.M, 1).transpose(2,1)
        dy = torch.stack([batch[:,1,:]]*self.M, 1).transpose(2,1)
        da = torch.stack([batch[:,2,:]]*self.M, 1).transpose(2,1)
        db = torch.stack([batch[:,3,:]]*self.M, 1).transpose(2,1)
        dslope = torch.stack([batch.data[:,4,:]] * self.M, 1).transpose(2,1)
        
        return [dx, dy, da, db, dslope]
    def bivariate_normal_pdf(self, point, parameter):
        dx, dy, da, db, dslope = [point[i] for i in range(len(point))]
        _, mu_x, mu_y, sigma_x, sigma_y, rho_xy, mu_a, mu_b, sigma_a, sigma_b, rho_ab, mu_slope, sigma_slope= [parameter[i] for i in range(len(parameter))]
        
        
        
        index_1=torch.isnan(mu_x).any()
        index_2=torch.isnan(mu_y).any()
        index_3=torch.isnan(mu_a).any()
        index_4=torch.isnan(mu_b).any()
        index_5=torch.isnan(mu_slope).any()
        
        if index_1:
            print("mu_x")
            print(mu_x)
        if index_2:
            print("mu_y")
            print(mu_y)
        if index_3:
            print("mu_a")
            print(mu_a)
        if index_4:
            print("mu_b")
            print(mu_b)
        if index_5:
            print("mu_slope")
            print(mu_slope)


        
        pi = torch.asin(torch.tensor(1.))*2

  

        z_x = ((dx-mu_x)/sigma_x)**2
        z_y = ((dy-mu_y)/sigma_y)**2
        z_xy = (dx-mu_x)*(dy-mu_y)/(sigma_x*sigma_y)
        Z_xy = z_x + z_y -2*rho_xy*z_xy
        exp_xy = torch.exp(-Z_xy/(2*(1-rho_xy**2)))
        norm_xy = 2*pi*sigma_x*sigma_y*torch.sqrt(1-rho_xy**2)
        result_xy=exp_xy/norm_xy
        
        z_a = ((da-mu_a)/sigma_a)**2
        z_b = ((db-mu_b)/sigma_b)**2
        z_ab = (da-mu_a)*(db-mu_b)/(sigma_a*sigma_b)
        Z_ab = z_a + z_b -2*rho_ab*z_ab
        exp_ab = torch.exp(-Z_ab/(2*(1-rho_ab**2)))
        norm_ab = 2*pi*sigma_a*sigma_b*torch.sqrt(1-rho_ab**2)
        result_ab=exp_ab/norm_ab

        z_slope=((dslope-mu_slope)/sigma_slope)**2
        exp_slope=torch.exp(-z_slope/2)
        norm_slope=torch.sqrt(2*pi)*sigma_slope
        result_slope=exp_slope/norm_slope

        return result_xy, result_ab, result_slope

    def reconstruction_loss(self, point, parameter):
       
        
        
        pdf_xy, pdf_ab, pdf_slope = self.bivariate_normal_pdf(point, parameter)
        
        
        
        batch_size=pdf_xy.shape[0]
        N_points=pdf_xy.shape[1]

        pi = parameter[0]
    

        LS_xy = -torch.sum(torch.log(1e-5+torch.sum(pi * pdf_xy, 2)))
        LS_ab = -torch.sum(torch.log(1e-5+torch.sum(pi * pdf_ab, 2)))
        LS_slope = -torch.sum(torch.log(1e-5+torch.sum(pi * pdf_slope, 2)))

        LS=(LS_xy+LS_ab+LS_slope)/(batch_size*N_points)
      
        return LS

    def forward(self,X):
        point = self.make_target(X.float())
        x, _, _=self.encoder(X.float())
        parameter=self.decoder(x)
        loss= self.reconstruction_loss(point,parameter)
        return loss

    def loss_on_loader(self, loader, device):
        # calculate loss on all data
        total = 0.0
        num = 0
        with torch.no_grad():
            for (i,inputs) in enumerate(loader):
                inputs = inputs.to(device)
                inputs = inputs.float()
               

                loss = self.forward(inputs)
                total += loss
                num += 1
        return total/num

