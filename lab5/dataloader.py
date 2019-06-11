from torch.utils.data.dataset import Dataset
import numpy as np
import string

def get_data(mode, alphabet):
    data = []
    if mode == 'train':
        with open('data/train.txt', 'r') as fo:
            w = fo.readlines()
            for line in w:
                word_list = line[:-1].split(' ')
                alpha_list = []
                for word in word_list:
                    temp = []
                    for alpha in word:
                        temp.append(alphabet[alpha])
                    alpha_list.append(np.array(temp))
                
                for i in range(4):
                    for j in range(4):
                        data.append([(i ,alpha_list[i]), (j, alpha_list[j])])
    elif mode == 'test':
        with open('data/test.txt', 'r') as fo:
            w = fo.readlines()
            for line in w:
                data.append(line[:-1].split(' '))
    return data

class TestLoader(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.alphabet = dict()
        for idx, alpha in enumerate(string.ascii_lowercase):
            self.alphabet[alpha] = idx + 2
        self.data = get_data(mode, self.alphabet)
        self.data_len = len(self.data)

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        data = []
        for word in self.data[idx][:2]:
            temp = []
            for alpha in word:
                temp.append(self.alphabet[alpha])
            data.append(np.array(temp))
        data.append(self.data[idx][2])
        data.append(self.data[idx][3])
        data = np.array(data)
        return data

class TrainLoader(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.alphabet = dict()
        for idx, alpha in enumerate(string.ascii_lowercase):
            self.alphabet[alpha] = idx + 2
        self.data = get_data(mode, self.alphabet)
        self.data_len = len(self.data)

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        return self.data[idx]

