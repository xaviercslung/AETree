import math
import numpy as np
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import sys
sys.path.append('../')

from model import *
import copy
import time
from tensorboardX import SummaryWriter
from dataloader import TreeData
import matplotlib.pyplot as plt



def sample_from_parameter(parameter, temperature=0.9):
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, mu_a, mu_b, sigma_a, sigma_b, rho_ab, mu_slope, sigma_slope= parameter

        def adjust_temp(pi_pdf,temperature):
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
        box_set=torch.zeros(64,5)
        
        for i in range(box_set.shape[0]):
            # get mixture indice:
            o_pi=pi[0,i,:]
            o_pi = adjust_temp(o_pi,temperature)
            pi_idx = torch.multinomial(o_pi, 1)
            
            # get mixture params:
            o_mu_x = mu_x[0,i,pi_idx]
            o_mu_y = mu_y[0,i,pi_idx]
            o_sigma_x = sigma_x[0,i,pi_idx]
            o_sigma_y = sigma_y[0,i,pi_idx]
            o_rho_xy = rho_xy[0,i,pi_idx]

            o_mu_a = mu_a[0,i,pi_idx]
            o_mu_b = mu_b[0,i,pi_idx]
            o_sigma_a = sigma_a[0,i,pi_idx]
            o_sigma_b = sigma_b[0,i,pi_idx]
            o_rho_ab = rho_ab[0,i,pi_idx]

            o_mu_slope = mu_slope[0,i,pi_idx]
            o_sigma_slope = sigma_slope[0,i,pi_idx]
        
            x,y = sample_bivariate_normal(o_mu_x,o_mu_y,o_sigma_x,o_sigma_y,o_rho_xy,temperature,greedy=False)
            a,b = sample_bivariate_normal(o_mu_a,o_mu_b,o_sigma_a,o_sigma_b,o_rho_ab,temperature,greedy=False)
            slope=sample_normal(o_mu_slope,o_sigma_slope,temperature)
            box_set[i,0]=x
            box_set[i,1]=y
            box_set[i,2]=a
            box_set[i,3]=b
            box_set[i,4]=slope
        return box_set


def sample(model, X,temperature):
    X=X.unsqueeze(0)
    num_points=X.shape[1]
    x, _, _=model.encoder(X.float())
    parameter=model.decoder(x)
    box=sample_from_parameter(parameter,temperature)
    return box


testset = TreeData(data_folder="./Data_pointnet_test.pickle")
test_loader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)

torch.cuda.set_device(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = pointnetbaseline.load_from_drive(pointnetbaseline, name='point_baseline_78000', model_dir='./log', device=device, n_feature=256)

X = next(iter(test_loader))

X = X.float()



box_total=[]
for i in range(X.shape[0]):
    for _ in range(10):
        box_set=sample(model,X[i,:,:],temperature=0.1*i+0.1)
        box_set=box_set.detach().numpy()
        box_total.append(box_set)


def rotate_xy(p, sin, cos, center):
    x_ = (p[:,0:1]-center[:,0:1])*cos-(p[:,1:2]-center[:,1:2])*sin+center[:,0:1]
    y_ = (p[:,0:1]-center[:,0:1])*sin+(p[:,1:2]-center[:,1:2])*cos+center[:,1:2]
#     print(((p[:,0:1]-center[:,0:1])*cos).shape, cos.shape, x_.shape)
    return np.hstack((x_, y_))
def get_box(P, F):
    ld = np.hstack((P[:,0:1]-F[:,0:1]/2, P[:,1:2]-F[:,1:2]/2))
    rd = np.hstack((P[:,0:1]+F[:,0:1]/2, P[:,1:2]-F[:,1:2]/2))
    ru = np.hstack((P[:,0:1]+F[:,0:1]/2, P[:,1:2]+F[:,1:2]/2))
    lu = np.hstack((P[:,0:1]-F[:,0:1]/2, P[:,1:2]+F[:,1:2]/2))
    # box = np.hstack((ld, rd, ru, lu)).reshape(len(P), -1, 2)
    sinO = F[:,2:3]
    cosO = np.cos(np.arcsin(F[:,2:3]))
    ld_r = rotate_xy(ld, sinO, cosO, P)
    rd_r = rotate_xy(rd, sinO, cosO, P)
    ru_r = rotate_xy(ru, sinO, cosO, P)
    lu_r = rotate_xy(lu, sinO, cosO, P)
    if(len(P)>0):
        box_r = np.hstack((ld_r, rd_r, ru_r, lu_r)).reshape(len(P), -1, 2)
    else:
        box_r = []
    return box_r

def convert_Euliean(X):
    x=X[0]
    y=X[1]
    a=X[2]
    b=X[3]
    degree=X[4]
    P=np.hstack((x,y))
    P=P.reshape(1,2)
    F=np.hstack((a,b,degree))
    F=F.reshape(1,3)
    box=get_box(P,F)
    point_1_x=box[0][0][0]
    point_2_x=box[0][1][0]
    point_3_x=box[0][2][0]
    point_4_x=box[0][3][0]

    point_1_y=box[0][0][1]
    point_2_y=box[0][1][1]
    point_3_y=box[0][2][1]
    point_4_y=box[0][3][1]
    poly=np.array([point_1_x,point_2_x,point_3_x,point_4_x,point_1_y,point_2_y,point_3_y,point_4_y])   
    return poly
def convert_ploy(x):
    polyset=[]
    for i in range(x.shape[0]):
        poly=convert_Euliean(x[i])
        polyset.append(poly)
    return polyset

def draw_poly(ax, pc):
    for i in range(len(pc)):
        X, Y= pc[i][0:4],pc[i][4:8]
#         ax.scatter(X, Y)
        ax.plot(X, Y,c='b')
        ax.plot([X[-1],X[0]], [Y[-1],Y[0]], c='b')
        ax.axis('equal')
        my_x_ticks = np.arange(-0.2, 1.2, 0.2)
        my_y_ticks = np.arange(-0.2, 1.2, 0.2)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)


def plot_polygon_noend(y,m=1,n=1, save=False, savename='pclouds'):
    fig = plt.figure(figsize=(15*m,15*n))
    fig.set_tight_layout(True)
   
    for i in range(n):
        for j in range(m):
            
            idx = i*m + j
            sample=convert_ploy(y)
            ax = fig.add_subplot(n, m, idx+1, xticks=[], yticks=[])
            draw_poly(ax, sample)
    if save:
        plt.savefig(savename, bbox_inches='tight', pad_inches=0)
        plt.close('all')
        plt.close(fig)


def subplot_num(m, i, j):
    return i*m + j

def plot_polygonset(samples, n, m, save=False, savename='pclouds'):
    fig = plt.figure(figsize=(10*m,10*n))
    fig.set_tight_layout(True)
    for i in range(n):
        for j in range(m):
            idx = subplot_num(m, i, j) 
#             print(i,j,m,n,idx)
            ax = fig.add_subplot(n, m, idx+1, xticks=[], yticks=[])
            
            o = samples[idx]
            index = np.arange(len(o))[o[:,2] == 1]
            if(len(index)>1):
#                 poly = trans_ori(o)
                poly=convert_ploy(o)
                draw_poly(ax, poly[:index[0]+1])
                for k in range(len(index)-1):
                    draw_poly(ax, poly[index[k]+1:index[k+1]+1])
            else:
#                 poly = trans_ori(o)
                poly=convert_ploy(o)
                draw_poly(ax, poly)
    if save:
        plt.savefig(savename, bbox_inches='tight', pad_inches=0)
    plt.show()


plot_polygonset(box_total, 10, 10)

