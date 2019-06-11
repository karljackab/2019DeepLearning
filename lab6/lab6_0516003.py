import torch
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np
import model
import torchvision.utils as utils

# Hyper parameter
Batch = 128
Random_noise_dim = 52
c_size = 2
Lr_d = 2e-4
Lr_g = 1e-3
Epochs = 101
device = torch.device('cuda:2')
##################################
c = np.linspace(-1, 1, 10).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)

c1 = np.hstack([c, np.zeros_like(c)])
c2 = np.hstack([np.zeros_like(c), c])

## Data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor()
        ])),
    batch_size=Batch, shuffle=True
)

## main
model_G = model.G().to(device).apply(model.weights_init)
model_D = model.D(c_size).to(device).apply(model.weights_init)
model_Q = model.Q(c_size).to(device).apply(model.weights_init)


opt_D = torch.optim.Adam(
    [
        {'params': model_D.parameters()}
    ]
    , betas=(0.5, 0.99), lr=Lr_d
)
opt_G = torch.optim.Adam(
    [
        {'params': model_G.parameters()},
        {'params': model_Q.parameters()}
    ], lr=Lr_g, betas=(0.5, 0.99)
)

D_loss = nn.BCELoss().cuda()
G_cross_loss = nn.CrossEntropyLoss().cuda()
def G_crit_loss(x, mu, var):
    log_loss = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    return log_loss.sum(1).mean().mul(-1.0)

def gen_random_noise(con_c, one_hot, batch_size):
    x = np.random.randn(batch_size, Random_noise_dim)
    x = torch.from_numpy(x).view(batch_size, Random_noise_dim, 1, 1).type(torch.FloatTensor)
    x = torch.cat((x, con_c, one_hot), 1).to(device)
    return x

def test(epoch):
    num = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]*10)
    one_hot = np.zeros((100, 10))
    one_hot[range(100), num] = 1
    dis_c = torch.from_numpy(one_hot)
    fix_noise = torch.from_numpy(np.random.randn(10, 52).repeat(10, axis = 0))

    con_c = torch.from_numpy(c1)
    z = torch.cat([fix_noise, dis_c, con_c], 1).view(-1, 64, 1, 1).type(torch.FloatTensor).to(device)
    fake = model_G(z)
    utils.save_image(fake.data, f'img/{epoch}_1.png', nrow=10)

    con_c = torch.from_numpy(c2)
    z = torch.cat([fix_noise, dis_c, con_c], 1).view(-1, 64, 1, 1).type(torch.FloatTensor).to(device)
    fake = model_G(z)
    utils.save_image(fake.data, f'img/{epoch}_2.png', nrow=10)

def train(y):
    opt_D.zero_grad()
    dis_loss = 0
    batch_size = y[1].shape[0]
    label = torch.from_numpy(np.random.randint(10, size=(batch_size, 1)))
    one_hot = torch.zeros(batch_size, 10).scatter_(1, label, 1).view(batch_size, 10, 1, 1)
    con_c = np.random.randn(batch_size, c_size)
    con_c = torch.from_numpy(con_c).view(batch_size, c_size, 1, 1).type(torch.FloatTensor)
    x = gen_random_noise(con_c, one_hot, batch_size)
    
    # Discriminator
    ## For Discriminator true term
    real_data = y[0].to(device)
    real_prob, _ = model_D(real_data)
    real_prob = real_prob.view(-1, 1)
    cls = torch.ones((batch_size, 1)).to(device)
    loss = D_loss(real_prob, cls)
    loss.backward()
    dis_loss += loss.data
    ## For Discriminator fake term
    fake = model_G(x)
    fake_prob, temp = model_D(fake.detach())
    cls = torch.zeros((batch_size, 1)).to(device)
    loss = D_loss(fake_prob.view(-1, 1), cls)
    loss.backward()
    dis_loss += loss.data
    opt_D.step()

    # Generator
    opt_G.zero_grad()
    gen_loss = 0
    ## For Generator fake term
    cls = torch.ones((batch_size, 1)).to(device)
    fake = model_G(x)
    fake_prob, temp = model_D(fake)
    gen_loss += D_loss(fake_prob.view(-1, 1), cls)
    ## For Q term
    fake_label, mu, var = model_Q(temp)
    gen_loss += G_cross_loss(fake_label, label.squeeze().to(device))
    con_c = con_c.to(device).view(mu.shape[0], mu.shape[1])
    gen_loss += G_crit_loss(con_c, mu, var)
    gen_loss.backward()
    opt_G.step()

    with torch.no_grad():
        real_prob = real_prob.mean()
        Q = G_cross_loss(fake_label, label.squeeze().to(device)) + G_crit_loss(con_c, mu, var)
        fake_prob_befor = fake_prob.mean()
        fake = model_G(x)
        fake_prob_after, _ = model_D(fake)
        fake_prob_after = fake_prob_after.mean()

    return dis_loss.data, gen_loss.data, Q.data, real_prob, fake_prob_befor, fake_prob_after

train_len = len(train_loader)
for epoch in range(Epochs):
    print(f'Epoch {epoch}')
    dis_train_avg, gen_train_avg, Q_avg = 0, 0, 0
    dis_total, gen_total, Q_total = 0, 0, 0
    cnt = 0
    rp, fpb, fpa = 0, 0, 0
    for iters, y in enumerate(train_loader):
        dis, gen, Q, real_prob, fake_prob_befor, fake_prob_after = train(y)
        dis_train_avg += dis
        gen_train_avg += gen - Q
        Q_avg += Q
        rp += real_prob
        fpb += fake_prob_befor
        fpa += fake_prob_after

        cnt += 1
        if iters%100 == 0:
            dis_total += dis_train_avg
            gen_total += gen_train_avg
            Q_total += Q_avg
            print(f'Epochs {epoch} {iters}/{train_len}: dis loss {dis_train_avg/cnt}, gen loss {gen_train_avg/cnt}, Q loss {Q_avg/cnt}')
            dis_train_avg, gen_train_avg, Q_avg = 0, 0, 0
            cnt = 0

    dis_total /= train_len
    gen_total /= train_len
    Q_total /= train_len
    rp /= train_len
    fpa /= train_len
    fpb /= train_len

    print(f'Epochs {epoch}: dis_avg loss {dis_total}, gen_avg loss {gen_total}, Q loss {Q_total}')
    print(f'real prob: {rp}, flase prob before: {fpb}, flase prob after: {fpa}')
    test(epoch)
    # write loss
    with open(f'dis_loss', 'a') as f:
        f.write(f'{str(float(dis_total))}\n')
    with open(f'gen_loss', 'a') as f:
        f.write(f'{str(float(gen_total))}\n')
    with open(f'Q_loss', 'a') as f:
        f.write(f'{str(float(Q_total))}\n')
    with open(f'rp', 'a') as f:
        f.write(f'{str(float(rp))}\n')
    with open(f'fpa', 'a') as f:
        f.write(f'{str(float(fpa))}\n')
    with open(f'fpb', 'a') as f:
        f.write(f'{str(float(fpb))}\n')

    # save model
    if epoch%10 == 0:
        torch.save(model_G, f'G_{str(epoch)}_me.pkl')