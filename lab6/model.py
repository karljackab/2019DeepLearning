import torch
import torch.nn as nn

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(64, 512, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        output = self.main(x)
        return output

class D(nn.Module):
    def __init__(self, c_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.discriminator = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        temp = self.main(x)
        
        RealFake = self.discriminator(temp)
        temp = temp.view(x.shape[0], -1)

        return RealFake, temp

class Q(nn.Module):
    def __init__(self, c_size):
        super().__init__()
        self.Q = nn.Sequential(
            nn.Linear(8192, 100, bias=True),
            nn.ReLU()
        )
        self.classify = nn.Linear(100, 10, bias=True)
        self.cal_mu = nn.Linear(100, c_size, bias=True)
        self.cal_var = nn.Linear(100, c_size, bias=True)
    def forward(self, x):
        temp = self.Q(x)
        label = self.classify(temp)
        mu = self.cal_mu(temp).squeeze()
        var = self.cal_var(temp).exp().squeeze()
        return label, mu, var

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
