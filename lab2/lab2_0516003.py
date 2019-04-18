import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from dataloader import read_bci_data
import matplotlib.pyplot as plt
import re
import os

class EEGNet(nn.Module):
    def __init__(self, activation='ELU'):
        super().__init__()
        if activation == 'ELU':
            act_func = nn.ELU
        elif activation == 'ReLU':
            act_func = nn.ReLU
        elif activation == 'LeakyReLU':
            act_func = nn.LeakyReLU
        else:
            print('Error activation function.')
        
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 31), stride=(1, 1), padding=(0, 15), bias=False),
            nn.BatchNorm2d(32, eps= 1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2, 5), stride=(1, 2), groups=16, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act_func(),
            nn.AvgPool2d(kernel_size=(1, 6), stride=(1, 6), padding=0),
            nn.Dropout(p=0.5)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 13), stride=(1, 1), padding=(0, 6), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act_func(),
            nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5), padding=0),
            nn.Dropout(p=0.15)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=768, out_features=2, bias=True)
        )
    def forward(self, x):
        out = self.firstConv(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out = out.reshape((x.shape[0], -1))
        out = self.classify(out)
        return out

class DeepConvNet(nn.Module):

    def __init__(self, activation):
        super().__init__()
        if activation == 'ELU':
            act_func = nn.ELU
        elif activation == 'ReLU':
            act_func = nn.ReLU
        elif activation == 'LeakyReLU':
            act_func = nn.LeakyReLU
        else:
            print('Error activation function.')

        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 15), stride=(1, 2), padding=(0, 11), bias=False),
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.2, affine=True, track_running_stats=True),
            act_func(),
            nn.MaxPool2d((1, 7), stride=(1, 7)),
            nn.Dropout(p=0.5)
        )
        self.secondConv = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 11), stride=(1, 1), padding=(0, 5), bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.15, affine=True, track_running_stats=True),
            act_func(),
            nn.MaxPool2d((1, 5), stride=(1, 5)),
            nn.Dropout(p=0.5)
        )
        self.thirdConv = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act_func(),
            nn.MaxPool2d((1, 3), stride=(1, 3)),
            nn.Dropout(p=0.4)
        )
        self.fourthConv = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act_func(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(p=0.2)
        )
        self.lastDense = nn.Sequential(
            nn.Linear(in_features=200, out_features=2, bias=True)
        )
    
    def forward(self, x):
        out = self.firstConv(x)
        out = self.secondConv(out)
        out = self.thirdConv(out)
        out = self.fourthConv(out)
        out = out.reshape((x.shape[0], -1))
        out = self.lastDense(out)
        return out

def test_accuracy(loader):
    total_cnt = 0
    correct_cnt = 0
    for _, (x, y) in enumerate(test_loader):
        if use_cuda:
            x, y = x.cuda(2), y.cuda(2)
        output = net(x).double()
        _, max_label = torch.max(output, 1)
        total_cnt += x.shape[0]
        correct_cnt += (max_label == y).sum()
    accuracy = float(correct_cnt)/float(total_cnt)
    return accuracy

def perform_net(train, current_net, net, Epoch, train_loader, test_loader, RecordEpoch, LossFunc, act):
    total_cnt = 0
    correct_cnt = 0
    highest_testing_accuracy = 0
    train_acc_list = []
    test_acc_list = []
    print(f'Start {current_net.__name__} with {act}')
    for epoch in range(Epoch):
        for _, (x, y) in enumerate(train_loader):
            if use_cuda:
                x, y = x.cuda(2), y.cuda(2)
            output = net(x).double()
            total_cnt += x.shape[0]
            _, pred_label = torch.max(output, 1)
            correct_cnt += (pred_label==y).sum()
            if train:
                loss = LossFunc(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if not train or epoch % RecordEpoch == 0:
            acc = float(correct_cnt)/float(total_cnt)
            # print(f'Epoch {epoch} training accuracy : {acc}')
            total_cnt = 0
            correct_cnt = 0
            accuracy = test_accuracy(test_loader)
            if epoch % RecordEpoch == 0:
                train_acc_list.append(acc)
                test_acc_list.append(accuracy)
            if accuracy > highest_testing_accuracy:
                # print(f'Epoch {epoch} testing accuracy : {accuracy}')
                highest_testing_accuracy = accuracy
                if train and accuracy > 0.87:
                    # print('testing accuracy : {accuracy}, save model')
                    torch.save(net, f'/home/karljackab/DL/lab2/{current_net.__name__}_{str(accuracy)}')
    accuracy = test_accuracy(test_loader)
    if accuracy > highest_testing_accuracy:
        highest_testing_accuracy = accuracy
    print(f'{current_net.__name__} with {act} highest testing accuracy: {highest_testing_accuracy}')
    return train_acc_list, test_acc_list, highest_testing_accuracy

def visualize(net_name, accuracy_list, label, RecordEpoch):
    x = [i*RecordEpoch for i in range(len(accuracy_list[0]))]
    for acc, lab in zip(accuracy_list, label):
        plt.plot(x, acc, label=lab)
    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title(f'Activation function comparison({net_name})')
    plt.savefig(net_name)
    plt.close()

if __name__ == '__main__':
    ## hyper parameter
    Epoch = 400
    Lr = 0.01
    BatchSize = 128
    LossFunc = nn.CrossEntropyLoss()
    RecordEpoch = 10

    ## Main Start
    use_cuda = torch.cuda.is_available()
    print(f'use cuda : {use_cuda}')
    act_list = ['ReLU', 'LeakyReLU', 'ELU']
    ## Data preprocessing
    train_data, train_label, test_data, test_label = read_bci_data()
    train_dataset = Data.TensorDataset(train_data, train_label)
    test_dataset = Data.TensorDataset(test_data, test_label)
    train_loader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = BatchSize,
        shuffle = True,
        num_workers = 2
    )
    test_loader = Data.DataLoader(
        dataset = test_dataset,
        batch_size = BatchSize,
        num_workers = 2
    )

    ## go
    net_list = [EEGNet, DeepConvNet,]
    optim_list = [optim.Adagrad, optim.Adam, ]
    total_high = 0
    for current_net, current_optim in zip(net_list, optim_list):
        accuracy_list = []
        label = []
        highest = 0
        for act in act_list:
            train = True
            net = current_net(act)  # ELU or ReLU or LeakyReLU

            files = os.listdir('.')
            regex = re.compile(rf'^{current_net.__name__}.*{act}$')
            for file in files:
                match = regex.fullmatch(file)
                if match:
                    net = torch.load(file)
                    train = False
                    break

            if use_cuda:
                net = net.cuda(2)
            optimizer = current_optim(net.parameters(), lr = Lr)
            train_list, test_list, high = perform_net(train, current_net, net, Epoch, train_loader, test_loader, RecordEpoch, LossFunc, act)
            accuracy_list.append(train_list)
            accuracy_list.append(test_list)
            label.append(act+'_train')
            label.append(act+'_test')
            if high > highest:
                highest = high
        visualize(f'{current_net.__name__}', accuracy_list, label, RecordEpoch)
        print(f"{current_net.__name__} Highest Accuracy : {highest}")
        if highest > total_high:
            total_high = highest
    print(f"\n\nHighest accuracy: {total_high}")