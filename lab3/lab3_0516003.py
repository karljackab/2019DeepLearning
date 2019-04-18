import utils.util as util
import utils.Dataload as loader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import utils.ResNet as ResNet
import utils.models as model
import os
import re
import torchvision.models as models

torch.cuda.set_device(2)

def load_net(net_name, pretrained = False):
    files = os.listdir('.')
    if pretrained:
        regex = re.compile(rf'^pre_{net_name}_.*pkl$')
    else:
        regex = re.compile(rf'^{net_name}_.*pkl$')
    for file in files:
        match = regex.fullmatch(file)
        if match:
            net = torch.load(file)
            break
    return net

def perform(net, optimizer, train_loader, test_loader, Epochs, LossFunc, net_name):  
    loading_batch_iter = 50
    high_acc = 0
    train_len = len(train_loader)
    test_len = len(test_loader)
    w_train_acc_list = []
    w_test_acc_list = []
    with open(f'pretrained_{net_name}', 'a') as f:
        for epoch in range(Epochs):
            total_cnt = 0
            correct_cnt = 0
            for idx, (x, y) in enumerate(train_loader):
                x = x.float()
                if use_cuda:
                    x, y = x.cuda(), y.cuda()
                output = net(x)
                total_cnt += x.shape[0]
                _, pred_label = torch.max(output, 1)
                correct_cnt += (pred_label==y).sum()
                loss = LossFunc(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx % loading_batch_iter == 0:
                    print(f'Epoch {epoch} Process {int(idx/loading_batch_iter)}/{int(train_len/loading_batch_iter)} ===>>> accuracy: {float(correct_cnt)/float(total_cnt)}')


            train_acc = float(correct_cnt)/float(total_cnt)
            w_train_acc_list.append(train_acc)
            print(f'Epoch {epoch} training accuracy : {train_acc}')

            total_cnt = 0
            correct_cnt = 0
            for idx, (x, y) in enumerate(test_loader):
                x = x.float()
                if use_cuda:
                    x, y = x.cuda(), y.cuda()
                output = net(x)
                total_cnt += x.shape[0]
                _, pred_label = torch.max(output, 1)
                correct_cnt += (pred_label==y).sum()
                if idx % loading_batch_iter == 0:
                    print(f'Process {int(idx/loading_batch_iter)}/{int(test_len/loading_batch_iter)}  ===>>> accuracy: {float(correct_cnt)/float(total_cnt)}')
            test_acc = float(correct_cnt)/float(total_cnt)
            f.write(f'{str(test_acc)}\n')
            w_test_acc_list.append(test_acc)
            print(f'Epoch {epoch} testing accuracy : {test_acc}')

            print('Save model')
            if test_acc > high_acc:
                torch.save(net, f'/home/karljackab/DL/lab3/{net_name}_{str(test_acc)}.pkl')
                high_acc = test_acc

    return [w_train_acc_list, w_test_acc_list]


## Hyper parameter
batch_size = 4
Lr = 0.001
Epochs = [10, 5]
nets = [ResNet.ResNet18(), ResNet.ResNet50()]
LossFunc = nn.CrossEntropyLoss()
pret_nets = [model.ResNet(18), model.ResNet(50)]

Load_model = True
Plot_confus = False

use_cuda = False
if torch.cuda.is_available():
    print("use Cuda")
    use_cuda = True

## data loader
train_data = loader.RetinopathyLoader('./data', 'train')
test_data = loader.RetinopathyLoader('./data', 'test')
train_loader = Data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, num_workers=2)
test_loader = Data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False, num_workers=2)
y_label = [train_data.get_label(), test_data.get_label()]

names = ["ResNet18", "ResNet50"]

if not Load_model:
    ## Main
    ### not pretrained
    ### idx=0->ResNet18, idx=1->ResNet50
    for i in range(1,2):
        net_name = names[i]
        net = nets[i]
        Epoch = Epochs[i]
        print(f'start {net_name} without pretrained')

        if use_cuda:
            net = net.cuda()

        optimizer = optim.SGD(net.parameters(), lr=Lr, momentum=0.9, weight_decay=0.0005)
        acc = perform(net, optimizer, train_loader, test_loader, Epoch, LossFunc, net_name)
        with open(f'{net_name}_train','w') as f:
            f.writelines(str(acc[0]))
        with open(f'{net_name}_test','w') as f:
            f.writelines(str(acc[1]))

    ### pretrained
    for i in range(0, 2):
        net_name = names[i]
        net = pret_nets[i]
        Epoch = Epochs[i]
        print(f'start {net_name} with pretrained')

        if use_cuda:
            net = net.cuda()
            
        optimizer = optim.SGD(net.parameters(), lr=Lr, momentum=0.9, weight_decay=0.0005)
        acc = perform(net, optimizer, train_loader, test_loader, Epoch, LossFunc, net_name)
        with open(f'pretrained_{net_name}_train','w') as f:
            f.writelines(str(acc[0]))
        with open(f'pretrained_{net_name}_test','w') as f:
            f.writelines(str(acc[1]))

else: ## evaluate stored model
    a = [False, True]
    for pretrained in a:
        for name in names:
            net = load_net(name, pretrained)
            if use_cuda:
                net = net.cuda()
            y_cnt = []
            pred_cnt = []
            total_cnt = 0
            correct_cnt = 0
            for x,y in test_loader:
                x = x.float()
                if use_cuda:
                    x, y = x.cuda(), y.cuda()
                output = net(x)
    
                total_cnt += x.shape[0]
                _, pred_label = torch.max(output, 1)
                if Plot_confus:
                    for y_true, o in zip(y, pred_label):
                        y_cnt.append(y_true.cpu())
                        pred_cnt.append(o.cpu())
                correct_cnt += (pred_label==y).sum()
            acc = float(correct_cnt)/float(total_cnt)
            if pretrained:
                print(f'pretrained {name} testing accuracy: {acc}')
                if Plot_confus:
                    util.plot_confusion(f'pretrained_{name}_testing_confusion', y_cnt, pred_cnt)
            else:
                print(f'{name} testing accuracy: {acc}')
                if Plot_confus:
                    util.plot_confusion(f'{name}_testing_confusion', y_cnt, pred_cnt)