from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import os
from torchvision import transforms
from PIL import Image

class RetinopathyLoader(Dataset):
    def __init__(self, root, mode, batch_size=1):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.batch_size = batch_size
        self.length = int(len(self.img_name))
        self.transform_0 = transforms.Compose([
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ])
        self.transform_n0 = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
        ])
        print(f">> Found {self.length} {mode} images")

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        label = self.label[idx]
        path = os.path.join(self.root, self.img_name[idx] + '.jpeg')
        img = Image.open(path).convert('RGB')
        if label == 0 or self.mode=='test':
            img = self.transform_0(img)
        else:
            img = self.transform_n0(img)
        return img, np.array(label)
    
    def get_label(self):
        return self.label

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
    return np.squeeze(img.values), np.squeeze(label.values)

