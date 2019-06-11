import matplotlib.pyplot as plt

folder = '32_const/'

test_bleu = []
train_KL_loss = []
train_loss = []

with open(folder + 'test_bleu') as f:
    for score in f.readlines():
        test_bleu.append(float(score.strip()))
with open(folder + 'train_KL_loss') as f:
    for score in f.readlines():
        s = score.split('(')[1].split(',')[0]
        train_KL_loss.append(float(s))
with open(folder + 'train_loss') as f:
    for score in f.readlines():
        train_loss.append(float(score.strip()))

x = range(1, len(test_bleu)+1)


plt.plot(x, train_KL_loss, label='train_KL_loss')
plt.plot(x, train_loss, label='train_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_fig.jpg')

plt.close()

plt.plot(x, test_bleu, label='test_bleu')
plt.xlabel('Epochs')
plt.ylabel('bleu score')
plt.legend()
plt.savefig('test_bleu.jpg')