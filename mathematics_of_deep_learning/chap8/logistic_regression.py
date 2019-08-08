import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn import svm

if __name__ == '__main__':
    set_matplotlib_formats('png', 'pdf')

    # sigmoid function graph
    xx = np.linspace(-6, 6, 500)
    yy = 1 / (np.exp(-xx) + 1)

    plt.figure(figsize=(6,6))
    plt.ylim(-3,3)
    plt.xlim(-3,3)
    plt.xticks(np.linspace(-3,3,13))
    plt.yticks(np.linspace(-3,3,13))
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.grid()
    plt.plot(xx, yy, c='b', label=r'$\dfrac{1}{1+\exp{(-x)}}$', lw=1)
    plt.plot(xx, xx, c='k', label=r'%y=x%', lw=1)
    plt.plot([-3,3],[0,0],c='k')
    plt.plot([0,0],[-3,3],c='k')
    plt.plot([-3,3],[1,1],linestyle='-.',c='k')
    plt.legend(fontsize=14)
    plt.show()

    # learning data
    iris = load_iris()
    x_org, y_org = iris.data, iris.target
    print('original data', x_org.shape, y_org.shape)

    x_data, y_data = iris.data[:100, :2], iris.target[:100]
    print('target data', x_data.shape, y_data.shape)

    x_data = np.insert(x_data, 0, 1.0, axis=1)
    print('added dummy variable', x_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=70, test_size=30, random_state=123)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # scatter plot of learning data
    x_t0 = x_train[y_train == 0]
    x_t1 = x_train[y_train == 1]
    plt.figure(figsize=(6,6))
    plt.scatter(x_t0[:,1],x_t0[:,2], marker='x', c='b', label='0 (setosa)')
    plt.scatter(x_t1[:,1],x_t1[:,2], marker='o', c='k', label='1 (versicolor)')
    plt.xlabel('sepal_length', fontsize=14)
    plt.ylabel('sepal_width', fontsize=14)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend(fontsize=16)
    plt.show()

    x_t0 = x_train[y_train == 0]
    x_t1 = x_train[y_train == 1]
    plt.figure(figsize=(6,6))
    plt.scatter(x_t0[:,1], x_t0[:,2], marker='x', s=50, c='b', label='yt=0')
    plt.scatter(x_t1[:,1], x_t1[:,2], marker='o', s=50, c='k', label='yt=1')
    plt.xlabel(r'$x_1$', fontsize=16)
    plt.ylabel(r'$x_2$', fontsize=16)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend(fontsize=16)
    plt.show()

    # variables
    x = x_train
    yt = y_train
    print('x[:5]', x[:5])
    print('yt[:5]', yt[:5])

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def pred(x, w):
        return sigmoid(x @ w)

    def cross_entropy(yt, yp):
        cel = - (yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
        return (np.mean(cel))

    def classify(y):
        return np.where(y < 0.5, 0, 1)

    def evaluate(xt, yt, w):
        yp = pred(xt, w)
        loss = cross_entropy(yt, yp)
        yp_b = classify(yp)
        score = accuracy_score(yt, yp_b)
        return loss, score

    # initialize
    M = x.shape[0]
    D = x.shape[1]
    iters = 10000
    alpha = 0.01
    w = np.ones(D)
    history = np.zeros((0,3))

    for k in range(iters):
        yp = pred(x, w)
        yd = yp - yt
        w = w - alpha * (x.T @ yd) / M

        if (k % 10 == 0):
            loss, score = evaluate(x_test, y_test, w)
            history = np.vstack((history, np.array([k, loss, score])))
            print( "iter = %d loss = %f score = %f" % (k, loss, score))

    print('initial state: loss: %f accracy: %f' % (history[0,1], history[0,2]))
    print('final state: loss: %f accracy: %f' % (history[-1,1], history[-1,2]))

    x_t0 = x_test[y_test == 0]
    x_t1 = x_test[y_test == 1]

    def b(x, w):
        return (-(w[0] + w[1] * x) / w[2])

    xl = np.asarray([x[:,1].min(), x[:,1].max()])
    yl = b(xl, w)

    plt.figure(figsize=(6,6))
    plt.scatter(x_t0[:,1], x_t0[:,2], marker='x', c='b', s=50, label='class 0')
    plt.scatter(x_t1[:,1], x_t1[:,2], marker='o', c='k', s=50, label='class 1')
    plt.plot(xl, yl, c='b')
    plt.xlabel('sepal_length', fontsize=14)
    plt.ylabel('sepal_width', fontsize=14)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend(fontsize=16)
    plt.show()

    # loss function
    plt.figure(figsize=(6,4))
    plt.plot(history[:,0], history[:,1], 'b')
    plt.xlabel('iter', fontsize=14)
    plt.ylabel('cost', fontsize=14)
    plt.title('iter vs cost', fontsize=14)
    plt.show()

    # accuracy
    plt.figure(figsize=(6,4))
    plt.plot(history[:,0], history[:,2], 'b')
    plt.xlabel('iter', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.title('iter vs accuracy', fontsize=14)
    plt.show()

    x1 = np.linspace(4, 7.5, 100)
    x2 = np.linspace(2, 4.5, 100)
    xx1, xx2 = np.meshgrid(x1, x2)
    xxx = np.asarray([np.ones(xx1.ravel().shape), xx1.ravel(), xx2.ravel()]).T
    c = pred(xxx, w).reshape(xx1.shape)
    plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1, projection='3d')
    ax.plot_surface(xx1, xx2, c, color='blue', edgecolor='black', rstride=10, cstride=10, alpha=0.1)
    ax.scatter(x_t1[:,1], x_t1[:,2], 1, s=20, alpha=0.9, marker='o', c='b')
    ax.scatter(x_t0[:,1], x_t0[:,2], 0, s=20, alpha=0.9, marker='s', c='b')
    ax.set_xlim(4, 7.5)
    ax.set_ylim(2, 4.5)
    ax.view_init(elev=20, azim=60)

    # sklearn library
    model_lr = LogisticRegression(solver='liblinear')
    model_svm = svm.SVC(kernel='linear')

    model_lr.fit(x, yt)
    model_svm.fit(x, yt)

    lr_w0 = model_lr.intercept_[0]
    lr_w1 = model_lr.coef_[0,1]
    lr_w2 = model_lr.coef_[0,2]
    svm_w0 = model_svm.intercept_[0]
    svm_w1 = model_svm.coef_[0,1]
    svm_w2 = model_svm.coef_[0,2]

    def rl(x):
        wk = lr_w0 + lr_w1 * x
        wk2 = - wk / lr_w2
        return wk2

    def svm(x):
        wk = svm_w0 + svm_w1 * x
        wk2 = - wk / svm_w2
        return wk2

    y_rl = rl(xl)
    y_svm = svm(xl)
    print(xl, yl, y_rl, y_svm)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    plt.scatter(x_t0[:,1], x_t0[:,2], marker='x', c='b')
    plt.scatter(x_t1[:,1], x_t1[:,2], marker='o', c='k')
    ax.plot(xl, yl, linewidth=2, c='k', label='Hands On')
    ax.plot(xl, y_rl, linewidth=2, c='k', linestyle='--', label='scikit LR')
    ax.plot(xl, y_svm, linewidth=2, c='k', linestyle='-.', label='scikit SVM')
    ax.legend()
    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    plt.show()

