import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as utils

def test(num):
    num_len = len(num)
    num = torch.tensor(num*10)
    one_hot = np.zeros((10*num_len, 10))
    one_hot[range(10*num_len), num] = 1
    dis_c = torch.from_numpy(one_hot)
    fix_noise = np.random.randn(num_len, 52)
    for i in range(num_len):
        fix_noise[i] = np.load(f'{num[i]}.npy')
    # np.save('a.npy', fix_noise)
    fix_noise = torch.from_numpy(fix_noise.repeat(10, axis = 0))
    
    c = np.linspace(0, 5, 10).reshape(1, -1)
    c = np.repeat(c, num_len, 0).reshape(-1, 1)

    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])

    con_c = torch.from_numpy(c1)
    z = torch.cat([fix_noise, dis_c, con_c], 1).view(-1, 64, 1, 1).type(torch.FloatTensor).to(device)
    fake = model(z)
    utils.save_image(fake.data, f'1.png', nrow=10)

    con_c = torch.from_numpy(c2)
    z = torch.cat([fix_noise, dis_c, con_c], 1).view(-1, 64, 1, 1).type(torch.FloatTensor).to(device)
    fake = model(z)
    utils.save_image(fake.data, f'2.png', nrow=10)

def loss_plot(path='store'):
    dl = []
    gl = []
    ql = []
    with open(f'{path}/dis_loss') as f:
        temp = f.readlines()
        for i in temp:
            dl.append(float(i[:-1]))
    with open(f'{path}/gen_loss') as f:
        temp = f.readlines()
        for i in temp:
            gl.append(float(i[:-1]))
    with open(f'{path}/Q_loss') as f:
        temp = f.readlines()
        for i in temp:
            ql.append(float(i[:-1]))
    x = range(len(temp))

    plt.plot(x, dl, label='discriminator loss')
    plt.plot(x, gl, label='generator loss')
    plt.plot(x, ql, label='Q loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss of model')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()

def prob_plot(path='store'):
    real = []
    fpb = []
    fpa = []
    with open(f'{path}/rp') as f:
        temp = f.readlines()
        for i in temp:
            real.append(float(i[:-1]))
    with open(f'{path}/fpb') as f:
        temp = f.readlines()
        for i in temp:
            fpb.append(float(i[:-1]))
    with open(f'{path}/fpa') as f:
        temp = f.readlines()
        for i in temp:
            fpa.append(float(i[:-1]))
    x = range(len(real))
    plt.plot(x, real, label='real data')
    plt.plot(x, fpb, label='fake data before update')
    plt.plot(x, fpa, label='fake data after update')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.title('Probability of data')
    plt.legend()
    plt.savefig('prob.png')
    plt.close()


# picture
weight = 'store/G_90_me.pkl'
device = torch.device('cuda:2')
model = torch.load(weight).to(device)
test([4])

## Chart
# loss_plot()
# prob_plot()