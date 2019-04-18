import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, down_sample, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.down_sample = down_sample
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample is not None:
            res = self.down_sample(x)
        else:
            res = x
        out += res
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        plane = [64, 128, 256, 512]
        class_num = 5
        self.conv1 = nn.Conv2d(3, plane[0], kernel_size=(7, 7),stride=2,bias=False)
        self.bn1 = nn.BatchNorm2d(plane[0])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        basic = [nn.Sequential(
                BasicBlock(plane[0], plane[0], None),
                BasicBlock(plane[0], plane[0], None)
            )]
        for i in range(1, 4):
            down_sample = nn.Sequential(
                nn.Conv2d(plane[i-1], plane[i], kernel_size=(1, 1), stride=2, bias=False),
                nn.BatchNorm2d(plane[i])
            )
            basic.append(nn.Sequential(
                BasicBlock(plane[i-1], plane[i], down_sample, 2),
                BasicBlock(plane[i], plane[i], None)
            ))
        self.layer1 = basic[0]
        self.layer2 = basic[1]
        self.layer3 = basic[2]
        self.layer4 = basic[3]

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.full = nn.Linear(plane[3], class_num)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size()[0], -1)
        out = self.full(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, down_sample, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=(1, 1), stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c*4, kernel_size=(1, 1), stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c*4)
        self.down_sample = down_sample
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.down_sample is not None:
            res = self.down_sample(x)
        else:
            res = x
        out += res
        out = self.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        plane = [64, 128, 256, 512]
        layer_num = [3, 4, 6, 3]
        class_num = 5
        self.conv1 = nn.Conv2d(3, plane[0], kernel_size=(7, 7),stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(plane[0])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)


        down_sample = nn.Sequential(
                nn.Conv2d(plane[0], 4*plane[0], kernel_size=(1, 1), stride=1, bias=False),
                nn.BatchNorm2d(4*plane[0])
            )
        basic = [nn.Sequential(
                Bottleneck(plane[0], plane[0], down_sample),
                Bottleneck(4*plane[0], plane[0], None),
                Bottleneck(4*plane[0], plane[0], None)
            )]
        for i in range(1, 4):
            down_sample = nn.Sequential(
                nn.Conv2d(4*plane[i-1], 4*plane[i], kernel_size=(1, 1), stride=2, bias=False),
                nn.BatchNorm2d(4*plane[i])
            )
            layers = [Bottleneck(4*plane[i-1], plane[i], down_sample, 2)]
            for _ in range(1, layer_num[i]):
                layers.append(Bottleneck(4*plane[i], plane[i], None))
            basic.append(nn.Sequential(*layers))

        self.layer1 = basic[0]
        self.layer2 = basic[1]
        self.layer3 = basic[2]
        self.layer4 = basic[3]

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.full = nn.Linear(4*plane[3], class_num)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size()[0], -1)
        out = self.full(out)
        return out