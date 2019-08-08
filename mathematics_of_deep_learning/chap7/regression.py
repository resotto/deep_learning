import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from sklearn.datasets import load_boston

if __name__ == '__main__':
    set_matplotlib_formats('png', 'pdf')

    boston = load_boston()
    x_org, yt = boston.data, boston.target
    print('debug: x_org', x_org)
    feature_names = boston.feature_names
    print('original data', x_org.shape, yt.shape)
    print('item name:', feature_names)

    x_data = x_org[:, feature_names == 'RM']
    print('after narrowing down', x_data.shape)

    x = np.insert(x_data, 0, 1.0, axis=1)

    # adding LSTAT
    x_add = x_org[:, feature_names == 'LSTAT']
    x2 = np.hstack((x, x_add))
    print(x2.shape)
    print(x2[:5, :])

#    print('after adding dummy variable', x.shape)
#    print('x', x[:5, :])
    print('yt', yt[:5])

    plt.scatter(x[:, 1], yt, s=10, c='b')
    plt.xlabel('ROOM', fontsize=14)
    plt.ylabel('PRICE', fontsize=14)
    plt.show()

    def pred(x, w):
        return(x @ w) # np.matmul

    # initialize
    M = x2.shape[0] # total number of input data
#    M = x.shape[0] # total number of input data
    D = x2.shape[1] # input dimensions
    print('D', D)
#    D = x.shape[1] # input dimensions
    iters = 2000
#    iters = 50000
    alpha = 0.001
#    alpha = 0.01
    w = np.ones(D)
    history = np.zeros((0,2))

    for k in range(iters):
        yp = pred(x2, w)
#        yp = pred(x, w)
        yd = yp - yt
        w = w - alpha * (x2.T @ yd) / M
#        w = w - alpha * (x.T @ yd) / M

        if (k % 100 == 0):
            loss = np.mean(yd ** 2) / 2
            history = np.vstack((history, np.array([k, loss])))
            print("iter = %d loss = %f" % (k, loss))

    print('initial value of loss function: %f' % history[0, 1])
    print('final value of loss function: %f' % history[-1, 1])

#    xall = x[:, 1].ravel()
#    xl = np.array([[1, xall.min()], [1, xall.max()]])
#    yl = pred(xl, w)
#
#    plt.figure(figsize=(6, 6))
#    plt.scatter(x[:, 1], yt, s=10, c='b')
#    plt.xlabel('ROOM', fontsize=14)
#    plt.ylabel('PRICE', fontsize=14)
#    plt.plot(xl[:, 1], yl, c='k')
#    plt.show()

    plt.plot(history[1:, 0], history[1:, 1])
    plt.show()

