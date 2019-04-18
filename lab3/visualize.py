import numpy as np
import utils.util as util

names_18 = ['ResNet18_train', 'ResNet18_test', \
        'pretrained_ResNet18_train', 'pretrained_ResNet18_test']

names_50 = ['ResNet50_train', 'ResNet50_test',\
        'pretrained_ResNet50_train', 'pretrained_ResNet50_test']

accs = []
for name in names_18:
    with open(name, 'r') as f:
        data = f.read()[1:-1].split(',')
    acc = [float(num) for num in data]
    accs.append(acc)
util.visualize('ResNet18', accs, names_18)

accs = []
for name in names_50:
    with open(name, 'r') as f:
        data = f.read()[1:-1].split(',')
    acc = [float(num) for num in data]
    accs.append(acc)
util.visualize('ResNet50', accs, names_50)
