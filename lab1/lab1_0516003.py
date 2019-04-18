import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        # distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Groud truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] < 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)

def loss(y_pred, y_true):
    return np.mean(((y_pred-y_true)**2)/2)
def der_loss(y_pred, y_true):
    return y_pred-y_true

# = = = = = = = = = =
class Sigmoid_Network():
    def __init__(self, layers):
        self.activation = sigmoid
        self.der_act = derivative_sigmoid
        self.loss = loss
        self.der_loss = der_loss
        self.weights = []

        l = len(layers)
        for i in range(1, l-1):
            self.weights.append(np.random.uniform(0, 1, (layers[i-1]+1, layers[i]+1)))  # plus bias term
        self.weights.append(np.random.uniform(0, 1, (layers[-2]+1, layers[-1])))
    
    def fit(self, X, y, Lr=0.1, epochs=100000, print_per_epoch=5000):
        ones = np.ones(len(X)).reshape(-1, 1)
        X = np.concatenate((X, ones), axis=1)

        for time in range(epochs):     # for every epochs
            for x,y_true in zip(X,y):         # for every single data
                # forward
                output = [x]
                for l_idx in range(len(self.weights)):
                    dot = output[l_idx].dot(self.weights[l_idx])
                    act = self.activation(dot)
                    output.append(act)
                output = np.array(output)

                # backpropagation
                deltas = []     # store deltas for weights update
                prev = []       # store previous gradient for backpropagation
                product = self.der_loss(output[-1], y_true)*self.der_act(output[-1])
                prev.append(product)
                delt = output[-2].reshape(-1, 1).dot(product.reshape(1, -1))
                deltas.append(delt)
                for i in range(len(output)-2, 0, -1):
                    product = self.weights[i].dot(prev[-1])*self.der_act(output[i])
                    prev.append(product)
                    delt = output[i-1].reshape(-1, 1).dot(product.reshape(1, -1))
                    deltas.append(delt)
                deltas.reverse()
                deltas = np.array(deltas)

                # update
                for i in range(len(self.weights)):
                    self.weights[i] -= Lr * deltas[i]

            if time % print_per_epoch == 0:
                x = X
                for l_idx in range(len(self.weights)):
                    x = self.activation(x.dot(self.weights[l_idx]))
                loss = self.loss(x, y)
                print(f'epoch {time} loss : {loss}')
                c = 1
                for i, j in zip(x, y):
                    if (i-0.5)*(j-0.5)<0:   ## predict y and true y are in different category
                        c=0
                        break
                if c==1:
                    print('All points are classified correctly, break training')
                    break
        
    def predict(self, X, y):
        ones = np.ones(len(X)).reshape(-1, 1)
        x = np.concatenate((X, ones), axis=1)
        for l_idx in range(len(self.weights)):
            x = self.activation(x.dot(self.weights[l_idx]))
        loss = self.loss(x, y)
        return x, loss

if __name__ == '__main__':
    x, y_true = generate_linear(100)
    # x, y_true = generate_XOR_easy()
    nn = Sigmoid_Network([2,4,4,1])
    nn.fit(x, y_true)
    y, _ = nn.predict(x, y_true)
    print(y)
    show_result(x, y_true, y)