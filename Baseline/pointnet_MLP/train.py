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
import os


def train_unsupervised(model, optimizer, scheduler, train_loader, test_loader, device, loss_save_dir,
                       num_epochs=100):
    print('Training your model!\n')
    model.train()

    best_params = None
    best_loss = float('inf')
    logs = defaultdict(list)
    
    try:
        for epoch in range(num_epochs):

            for (i,inputs) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = inputs.to(device)
                inputs = inputs.float() 
                loss = model(inputs)
                
                loss.backward()
                
                lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = max(lr, 1e-6)
                optimizer.step()
            scheduler.step()
            
            if (epoch % 1) == 0 or epoch == num_epochs-1:
                start_time = time.time()
                train_loss = model.loss_on_loader(train_loader, device)
                end_time = time.time()
                train_time = end_time - start_time
                
                start_time = time.time()
                test_loss = model.loss_on_loader(test_loader, device)
                end_time = time.time()
                test_time = end_time - start_time
                
                logs['train_loss'].append(train_loss)
                logs['test_loss'].append(test_loss)
                
                writer.add_scalars('point_baseline',
                                   {'train_loss': train_loss,
                                    'test_loss': test_loss,
                                   }, epoch)
                
                if (torch.isnan(test_loss)):
                    print("Epoch {epoch} Loss in nan!!!".format(epoch=epoch))
                    break;
                else:
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_params = copy.deepcopy(model.state_dict())

                    print("Epoch {epoch} Lr={Lr}, train loss={train_loss}, traintime={train_time}, test loss={test_loss}, test time={test_time}"
                           .format(epoch=epoch, Lr= lr, train_loss=train_loss, train_time=train_time, test_loss=test_loss, test_time=test_time)
                         )
            if (epoch % 1000) == 0 or epoch == num_epochs-1:
                model.save_to_drive(name=model.DEFAULT_SAVED_NAME+"_"+str(epoch))

    except KeyboardInterrupt:
        pass
#     print(best_loss, best_params)
    model.load_state_dict(best_params)
#     print(model.state_dict())
    model.eval()
    model.cpu()

    print('Saving model to drive...', end='')
    model.save_to_drive()
    print('done.')

    print('Saving training logs...', end='')
    np_logs = np.stack([np.array(item) for item in logs.values()], axis=0)
    if not os.path.exists(loss_save_dir):
        os.makedirs(loss_save_dir)
    np.save(loss_save_dir+model.DEFAULT_SAVED_NAME, np_logs)
    print('done.')


def train_ae(model, train_loader, test_loader, device, loss_save_dir, num_epochs=1000):
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
#     test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, drop_last=True)

    optimizer = Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.5)

    train_unsupervised(model, optimizer, scheduler, train_loader, test_loader, device, loss_save_dir,
                        num_epochs=num_epochs)



trainset = TreeData(data_folder="./Data_pointnet_train.pickle")
train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

testset = TreeData(data_folder="./Data_pointnet_test.pickle")
test_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False)



torch.cuda.set_device(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

loss_save_dir = './log/'
writer = SummaryWriter('point_baseline')
model = pointnetbaseline(device, save_name='point_baseline', n_feature=256)
model.to(device)
train_ae(model, train_loader, test_loader, device, loss_save_dir, num_epochs=100000)
writer.export_scalars_to_json("./point_baseline.json")
writer.close()


