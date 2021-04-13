import math
import numpy as np
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import sys

sys.path.append('../')

from model_ae_tree_box_ab2_new_re_weight_lstm_print import *
import copy
import time
from tensorboardX import SummaryWriter

try:
    weightLeaf, weightType, learningRate, testNum, input_data_name, numBox = sys.argv[1:]
    weightLeaf = float(weightLeaf)
    weightType = int(weightType)
    learningRate = float(learningRate)
    testNum = 'test_' + str(testNum)
    input_data = '/home/sc8635/' + input_data_name
    numBox = int(numBox)
except:
    print('invalid number of arguments')

print('weight leaf: ' + str(weightLeaf))
print('weight type: ' + str(weightType))
print('learning rate: ' + str(learningRate))
print('test number: ' + str(testNum))
print('input data DIR: ' + str(input_data))
print('number of boxes: ' + str(numBox))


def train_unsupervised(model, optimizer, scheduler, train_loader, test_loader, device, loss_save_dir,
                       num_epochs=100, M=1):
    print('Training your model!\n')
    model.train()

    best_params = None
    best_loss = float('inf')
    logs = defaultdict(list)

    try:
        for epoch in range(num_epochs):

            train_loss = 0
            train_loss_ab = 0
            #             train_loss_ov = 0
            train_loss_p = 0
            train_loss_leaf = 0

            train_loss_left_check = np.zeros((20, 1))
            train_loss_right_check = np.zeros((20, 1))

            num = 0
            start_time = time.time()
            for i, (node_xys, I_list, node_fea, node_is_leaf) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                #                 print(i)
                node_xys = node_xys.to(device)
                node_xys = node_xys.float()
                I_list = [t.to(device) for t in I_list]
                node_fea = node_fea.to(device)
                node_fea = node_fea.float()
                node_is_leaf = node_is_leaf.to(device)

                loss, loss_ab, loss_p, loss_leaf, left_check, right_check = model(node_xys, node_fea, I_list,
                                                                                  node_is_leaf)
                loss.backward()

                for i, ITEM in enumerate(left_check):
                    train_loss_left_check[i] += ITEM.item()
                for i, ITEM in enumerate(right_check):
                    train_loss_right_check[i] += ITEM.item()
                #                 for name, parms in model.named_parameters():
                #                     print('-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad,' -->leaf:', parms.is_leaf)

                #                 torch.nn.utils.clip_grad_norm(model.parameters(), 5)

                train_loss += loss.item()
                train_loss_ab += loss_ab
                #                 train_loss_ov += loss_overlap
                train_loss_p += loss_p.item()
                train_loss_leaf += loss_leaf
                num += 1

                lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = max(lr, 1e-6)

                optimizer.step()
            scheduler.step()

            train_loss = train_loss / num
            train_loss_ab = train_loss_ab / num
            #             train_loss_ov = train_loss_ov/num
            train_loss_p = train_loss_p / num
            train_loss_leaf = train_loss_leaf / num
            train_loss_left_check = train_loss_left_check / num
            train_loss_right_check = train_loss_right_check / num

            LOG_loss.write('Epoch: %d\n' % (epoch + 1))
            LOG_loss.write(f'Train_Loss_Left: {train_loss_left_check}\nTrain_Loss_right: {train_loss_right_check}\n')
            LOG_loss.flush()

            end_time = time.time()
            train_time = end_time - start_time
            if (epoch % 1) == 0 or epoch == num_epochs - 1:
                #                 start_time = time.time()
                #                 train_loss, train_loss_ab, train_loss_p, train_loss_leaf = model.loss_on_loader(train_loader, device)
                #                 end_time = time.time()
                #                 train_time = end_time - start_time

                start_time = time.time()
                test_loss, test_loss_ab, test_loss_p, test_loss_leaf, test_loss_left_check, test_loss_right_check = model.loss_on_loader(
                    test_loader, device)

                LOG_loss.write(f'Test_Loss_Left: {test_loss_left_check}\nTest_Loss_right: {test_loss_right_check}\n')
                LOG_loss.flush()

                end_time = time.time()
                test_time = end_time - start_time

                logs['train_loss'].append(train_loss)
                logs['test_loss'].append(test_loss)

                writer.add_scalars('ae_lstm_' + str(numBox) + '_' + testNum,
                                   {'train_loss_' + testNum: train_loss,
                                    'train_loss_ab_' + testNum: train_loss_ab,
                                    'train_loss_p_' + testNum: train_loss_p,
                                    'train_loss_leaf_' + testNum: train_loss_leaf,
                                    'test_loss_' + testNum: test_loss,
                                    'test_loss_ab_' + testNum: test_loss_ab,
                                    'test_loss_p_' + testNum: test_loss_p,
                                    'test_loss_leaf_' + testNum: test_loss_leaf, }, epoch)

                if (torch.isnan(test_loss)):
                    print("Epoch {epoch} Loss in nan!!!".format(epoch=epoch))
                    break;
                else:
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_params = copy.deepcopy(model.state_dict())

                        model.save_to_drive(name=model.DEFAULT_SAVED_NAME + "_best")
                        print("Epoch {epoch} model saving ...".format(epoch=epoch))

                    print(
                        "Epoch {epoch} Lr={Lr}, train loss={train_loss}, traintime={train_time}, test loss={test_loss}, test time={test_time}"
                            .format(epoch=epoch, Lr=lr, train_loss=train_loss, train_time=train_time,
                                    test_loss=test_loss,
                                    test_time=test_time)
                    )
            if (epoch % 200) == 0 or epoch == num_epochs - 1:
                model.save_to_drive(name=model.DEFAULT_SAVED_NAME + "_" + str(epoch))

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
    np_logs = np.stack([np.array(torch.tensor(item)) for item in logs.values()], axis=0)
    if not os.path.exists(loss_save_dir):
        os.makedirs(loss_save_dir)
    np.save(loss_save_dir + model.DEFAULT_SAVED_NAME, np_logs)
    print('done.')

    LOG_loss.close()


def train_ae(model, train_loader, test_loader, device, loss_save_dir, learningRate, M=1, num_epochs=1000, ):
    #     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    #     test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, drop_last=True)

    optimizer = Adam(model.parameters(), lr=learningRate)
    scheduler = StepLR(optimizer, step_size=400, gamma=0.5)

    train_unsupervised(model, optimizer, scheduler, train_loader, test_loader, device, loss_save_dir,
                       M=M, num_epochs=num_epochs)


trainset = TreeData(data_folder="/home/sc8635/AETree_64.pickle", train=True, split=0.9, n_feature=512, num_box=numBox,
                    batch_size=50)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

testset = TreeData(data_folder="/home/sc8635/AETree_64.pickle", train=False, split=0.9, n_feature=512, num_box=numBox,
                   batch_size=50)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

torch.cuda.set_device(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

loss_save_dir = './log/'
if not os.path.exists(loss_save_dir):
    os.makedirs(loss_save_dir)
LOG_loss = open(os.path.join(loss_save_dir, 'tree_lstm_64_log_loss'+testNum+'.txt'), 'w')
writer = SummaryWriter(testNum)
model = AE(device, leaf_loss=True, weight_leaf=weightLeaf, weight_type=weightType,
           save_name='tree_lstm_64'+testNum, n_feature=512)
model.to(device)
train_ae(model, train_loader, test_loader, device, loss_save_dir, learningRate=learningRate, num_epochs=3000, M=1, )
writer.export_scalars_to_json("./tree_lstm_64"+testNum+".json")
writer.close()
