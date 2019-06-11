import numpy as np
import matplotlib.pyplot as plt

def get_data(bits=8):
    x = np.random.randint(-128, 128)
    x_num = np.binary_repr(x, width=bits)
    x_num = [int(i) for i in x_num]

    y = np.random.randint(-128, 128)
    y_num = np.binary_repr(y, width=bits)
    y_num = [int(i) for i in y_num]

    num = [[x, y] for x, y in zip(x_num, y_num)]

    res = x + y
    res_num = np.binary_repr(res, width=bits)
    res_num = np.array([int(i) for i in res_num][-1:-9:-1])[::-1]

    # num is a zip list, ex: x=1, y=2 => [[0, 0], [0, 0], ... [0, 1], [1, 0]]
    # res_num is a numpy array, ex: x=1, y=2 => res=3 => [0, 0, ... ,1, 1]
    return num, res_num


class RNN_binAddModel():
    def __init__(self):
        ## init weight and gradient
        self.latent = 8
        self.wx, self.wo, self.ww =  [], [], []
        self.bx, self.co = [], []
        for _ in range(8):
            self.wx.append(np.random.rand(self.latent, 2))
            self.bx.append(np.random.rand(self.latent, 1))

            self.wo.append(np.random.rand(1, self.latent))
            self.co.append(np.random.rand(1, 1))
            self.ww.append(np.random.rand(self.latent, self.latent))

    def forward(self, x, y, Lr):
        hidden = []
        output = []
        x, y = x[::-1], y[::-1]
        for idx, x_unit in enumerate(x):
            x[idx] = np.atleast_2d(x_unit).reshape(2, 1)
        for idx, y_unit in enumerate(y):
            y[idx] = np.atleast_2d(y_unit)
        a = self.bx[0] + self.wx[0].dot(x[0])
        hidden.append(np.tanh(a))
        o = self.co[0] + self.wo[0].dot(hidden[0])
        output.append(o)
        for i in range(1, 8):
            a = self.bx[i] + self.ww[i].dot(hidden[i-1]) + self.wx[i].dot(x[i])
            hidden.append(np.tanh(a))
            o = self.co[i] + self.wo[i].dot(hidden[i])
            output.append(o)
        
        self.backward(x, y, hidden, output, Lr)

        res = []
        for i in output:
            if i[0][0] > 0.5:
                res.append(1)
            else:
                res.append(0)
        res = np.array(res)[::-1]
        return res

    def backward(self, x, y, hidden, output, Lr):
        h_g = []
        loss_g = []
        wx_g = []
        ww_g = []
        wo_g = []
        bx_g = []
        co_g = []
        H = []

        # preprocess H and gradient of loss
        for i in range(8):
            H.append(np.diag(1-np.power(hidden[i].reshape(self.latent), 2)))
            loss_g.append(output[i] - y[i])

        # gradient of h
        h_g.append(self.wo[-1].transpose().dot(loss_g[-1]))
        for i in range(6, -1, -1):
            temp = self.ww[i].transpose().dot(H[i+1]).dot(h_g[-1])
            h_g.append(temp + self.wo[i].transpose().dot(loss_g[i]))
        h_g = h_g[::-1]

        # gradient of ww
        ww_g.append(np.zeros((self.latent, self.latent)))
        for i in range(1, 8):
            temp = H[i].dot(h_g[i]).dot(hidden[i-1].transpose())
            ww_g.append(temp)

        # gradient of wx, wo and bx
        for i in range(8):
            wx_g.append(H[i].dot(h_g[i]).dot(x[i].transpose()))
            wo_g.append(loss_g[i].dot(hidden[i].transpose()))
            bx_g.append(H[i].dot(h_g[i]))
            co_g.append(loss_g[i])

        # update weight
        for i in range(8):
            self.wx[i] -= Lr*wx_g[i]
            self.wo[i] -= Lr*wo_g[i]
            self.ww[i] -= Lr*ww_g[i]
            self.bx[i] -= Lr*bx_g[i]
            self.co[i] -= Lr*co_g[i]


def visualize(acc, iteration, record):
    x = [i*record for i in range(int(iteration/record) + 1)]
    acc = [0] + acc
    plt.plot(x, acc)

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy(%)')
    plt.title(f'Handmade RNN accuracy')
    plt.savefig(f'lab4_acc')
    plt.close()

## Hyper parameter
Lr = 0.1
Iteration = 30000
Record_cnt = 1000

model = RNN_binAddModel()

cor_cnt = 0
err_dig = 0
accs = []
for iter in range(Iteration):
    x, y = get_data()
    res = model.forward(x, y, Lr)
    comp = 8 - (y==res).sum()
    err_dig += comp
    if comp == 0:
        cor_cnt += 1
    
    if iter % Record_cnt == 0:
        print(f'{iter:5} Error: {err_dig/Record_cnt/8}')
        print(f'{iter:5} Accuracy: {cor_cnt/Record_cnt}')
        accs.append(cor_cnt/Record_cnt)
        cor_cnt = 0
        err_dig = 0
visualize(accs, Iteration, Record_cnt)