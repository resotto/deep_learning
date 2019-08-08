import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
from IPython.display import set_matplotlib_formats
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MNIST_FILE = 'mnist-original.mat'
MNIST_PATH = 'mldata'
MNIST_URL = 'https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat'

if __name__ == '__main__':
    set_matplotlib_formats('png', 'pdf')

    mnist_fullpath = os.path.join('.', MNIST_PATH, MNIST_FILE)
    if not os.path.isfile(mnist_fullpath):
        mldir = os.path.join('.', 'mldata')
        os.makedirs(mldir, exist_ok=True)
        print('download %s started.' % MNIST_FILE)
        urllib.request.urlretrieve(MNIST_URL, mnist_fullpath)
        print('download %s finished.' % MNIST_FILE)

    mnist = fetch_mldata('MNIST original', data_home='.')
    x_org, y_org = mnist.data, mnist.target

    x_norm = x_org / 255.0
    x_all = np.insert(x_norm, 0, 1, axis=1)
    print('added dummy variable', x_all.shape)

    ohe = OneHotEncoder(sparse=False)
    y_all_one = ohe.fit_transform(np.c_[y_org])
    print('after one hot vectorization', y_all_one.shape)

    x_train, x_test, y_train, y_test, y_train_one, y_test_one = \
        train_test_split(x_all, y_org, y_all_one, train_size = 60000, test_size=10000, shuffle=False)
    print('x_train.shape, x_test.shape, y_train.shape, y_test.shape, t_train_one.shape, y_test_one.shape')
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, y_train_one.shape, y_test_one.shape)

    N = 20
    np.random.seed(123)
    indices = np.random.choice(y_test.shape[0], N, replace=False)
    x_selected = x_test[indices, 1:]
    y_selected = y_test[indices]
    plt.figure(figsize=(10,3))
    for i in range(N):
        ax = plt.subplot(2, N/2, i + 1)
        plt.imshow(x_selected[i].reshape(28,28),cmap='gray_r')
        ax.set_title('%d' % y_selected[i], fontsize=16)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def softmax(x):
        x = x.T
        x_max = x.max(axis=0)
        x = x - x_max
        w = np.exp(x)
        return (w / w.sum(axis=0)).T

    def ReLU(x):
        return np.maximum(0, x)

    def step(x):
        return 1.0 * (x > 0)

    def cross_entropy(yt, yp):
        return -np.mean(np.sum(yt * np.log(yp), axis=1))

    def evaluate(x_test, y_test, y_test_one, V, W):
        b1_test = np.insert(sigmoid(x_test @ V), 0, 1, axis=1)
        yp_test_one = softmax(b1_test @ W)
        yp_test = np.argmax(yp_test_one, axis=1)
        loss = cross_entropy(y_test_one, yp_test_one)
        score = accuracy_score(y_test, yp_test)
        return score, loss

    def evaluate_with_ReLU(x_test, y_tesst, y_test_one, V, W):
        b1_test = np.insert(ReLU(x_test @ V), 0, 1, axis=1)
        yp_test_one = softmax(b1_test @ W)
        yp_test = np.argmax(yp_test_one, axis=1)
        loss = cross_entropy(y_test_one, yp_test_one)
        score = accuracy_score(y_test, yp_test)
        return score, loss

    def evaluate_more_hiddenlayer(x_test, y_test, y_test_one, U, V, W):
        b1_test = np.insert(ReLU(x_test @ U), 0, 1, axis=1)
        d1_test = np.insert(ReLU(b1_test @ V), 0, 1, axis=1)
        yp_test_one = softmax(d1_test @ W)
        yp_test = np.argmax(yp_test_one, axis=1)
        loss = cross_entropy(y_test_one, yp_test_one)
        score = accuracy_score(y_test, yp_test)
        return score, loss

    class Indices():
        '''
        class which fetches index for mini-batch
        '''
        def __init__(self, total, size):
            self.total = total
            self.size = size
            self.indexes = np.zeros(0)

        def next_index(self):
            next_flag = False

            if len(self.indexes) < self.size:
                self.indexes = np.random.choice(self.total, self.total, replace=False)
                next_flag = True

            index = self.indexes[:self.size]
            self.indexes = self.indexes[self.size:]
            return index, next_flag

    # Indices class test
#    indices = Indices(20, 5)
#    for i in range(6):
#        arr, flag = indices.next_index()
#        print('arr, flag')
#        print(arr, flag)

    # ReLU and step function graph
    xx = np.linspace(-4, 4, 501)
    yy = ReLU(xx)
    plt.figure(figsize=(6,6))
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    plt.grid(lw=2)
    plt.plot(xx, ReLU(xx), c='b', label='ReLU', linestyle='-', lw=3)
    plt.plot(xx, step(xx), c='k', label='step', linestyle='-.', lw=3)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.legend(fontsize=14)
    plt.show()


    H = 128
    H1 = H + 1
    M = x_train.shape[0]
    D = x_train.shape[1]
    N = y_train_one.shape[1]

    nb_epoch = 200
    batch_size = 512
    alpha = 0.01
    U = np.random.randn(D, H) / np.sqrt(D / 2)
#    V = np.ones((D,H))
    V = np.random.randn(H1, H) / np.sqrt(H1 / 2)
#    W = np.ones((H1,N))
    W = np.random.randn(H1, N) / np.sqrt(H1 / 2)
    print('V[:2,:5]')
    print(V[:2,:5])
    print('W[:2,:5]')
    print(W[:2,:5])
    history1 = np.zeros((0,3))
    indices = Indices(M, batch_size)
    epoch = 0

    while epoch < nb_epoch:
        index, next_flag = indices.next_index()
        x, yt = x_train[index], y_train_one[index]

#        a = x @ V
        a = x @ U
#        b = sigmoid(a)
        b = ReLU(a)
        b1 = np.insert(b, 0, 1, axis=1)
#        u = b1 @ W
        c = b1 @ V
        d = ReLU(c)
        d1 = np.insert(d, 0, 1, axis=1)
#        y = d1 @ W
        u = d1 @ W
        yp = softmax(u)

        yd = yp - yt
#        bd = b * (1-b) * (yd @ W[1:].T)
#        bd = step(a) * (yd @ W[1:].T)
        dd = step(c) * (yd @ W[1:].T)
        bd = step(a) * (dd @ V[1:].T)

        W = W - alpha * (d1.T @ yd) / batch_size
#        W = W - alpha * (b1.T @ yd) / batch_size
        V = V - alpha * (b1.T @ dd) / batch_size
        U = U - alpha * (x.T @ bd) / batch_size

        if next_flag:
            score, loss = evaluate_more_hiddenlayer(x_test, y_test, y_test_one, U, V, W)
            history1 = np.vstack((history1, np.array([epoch, loss, score])))
            print('epoch = %d loss = %f score = %f' % (epoch, loss, score))
            epoch = epoch + 1

    print('initial state: loss: %f, accuracy: %f' % (history1[0,1], history1[0,2]))
    print('final state: loss: %f, accuracy: %f' % (history1[-1,1], history1[-1,2]))

    # loss graph
    plt.plot(history1[:,0], history1[:,1])
    plt.ylim(0,2.5)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.show()

    # accuracy graph
    plt.plot(history1[:,0], history1[:,2])
    plt.ylim(0,1)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.show()

    # prediction and answer check
    N = 20
    np.random.seed(123)
    indices = np.random.choice(y_test.shape[0], N, replace=False)
    x_selected = x_test[indices]
    y_selected = y_test[indices]

#    b1_test = np.insert(ReLU(x_selected @ V), 0, 1, axis=1)
    b1_test = np.insert(ReLU(x_selected @ U), 0, 1, axis=1)
    d1_test = np.insert(ReLU(b1_test @ V), 0, 1, axis=1)
#    yp_test_one = softmax(b1_test @ W)
    yp_test_one = softmax(d1_test @ W)
    yp_test = np.argmax(yp_test_one, axis=1)

    plt.figure(figsize=(10,3))
    for i in range(N):
        ax = plt.subplot(2, N/2, i + 1)
        plt.imshow(x_selected[i,1:].reshape(28,28),cmap='gray_r')
        ax.set_title('%d:%d' % (y_selected[i], yp_test[i]), fontsize=14)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()





























































































