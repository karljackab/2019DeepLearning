from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion(file_name, y_true, y_pred, normalize=True, title=None):
    np.set_printoptions(precision=2)
    comMat= confusion_matrix(y_true, y_pred)
    # classes = unique_labels(y_true, y_pred)
    classes = [0,1,2,3,4]
    comMat = comMat.astype('float') / comMat.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(comMat, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(comMat.shape[1]),
           yticks=np.arange(comMat.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    fmt = '.2f'
    thresh = comMat.max() / 2.
    for i in range(comMat.shape[0]):
        for j in range(comMat.shape[1]):
            ax.text(j, i, format(comMat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if comMat[i, j] > thresh else "black")
    fig.tight_layout()

    plt.savefig(file_name)
    plt.close()

def visualize(net_name, accuracy_list, label):
    x = [i for i in range(len(accuracy_list[0]))]
    for acc, lab in zip(accuracy_list, label):
        plt.plot(x, acc, label=lab)
    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title(f'Result Comparison({net_name})')
    plt.savefig(f'{net_name}_acc_result')
    plt.close()