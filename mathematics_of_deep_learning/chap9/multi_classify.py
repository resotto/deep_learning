import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

if __name__ == '__main__':
    set_matplotlib_formats('png', 'pdf')

    iris = load_iris()
    x_org, y_org = iris.data, iris.target
    x_select = x_org[:,[0,2]]
    print('original data', x_select.shape, y_org.shape)

    # scatter graph
    x_t0 = x_select[y_org == 0]
    x_t1 = x_select[y_org == 1]
    x_t2 = x_select[y_org == 2]
    plt.figure(figsize=(6,6))
    plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='k', s=50, label='0 (setosa)')
    plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='b', s=50, label='1 (versicolour)')
    plt.scatter(x_t2[:,0], x_t2[:,1], marker='+', c='k', s=50, label='2 (verginica)')
    plt.xlabel('sepal_length', fontsize=14)
    plt.ylabel('petal_length', fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.legend(fontsize=14)
    plt.show()

    x_all = np.insert(x_select, 0, 1.0, axis=1)

    # changing to 4 dimensions
    x_all2 = np.insert(x_org, 0, 1.0, axis=1)

    ohe = OneHotEncoder(sparse=False, categories='auto')
    y_work = np.c_[y_org]
    y_all_one = ohe.fit_transform(y_work)
    print('original', y_org.shape)
    print('2 dimensionalization', y_work.shape)
    print('One Hot Vectorization', y_all_one.shape)

#    x_train, x_test, y_train, y_test, y_train_one, y_test_one = \
#        train_test_split(x_all, y_org, y_all_one, train_size=75, test_size=75, random_state=123)
    x_train2, x_test2, y_train, y_test, y_train_one, y_test_one = \
        train_test_split(x_all2, y_org, y_all_one, train_size=75, test_size=75, random_state=123)
#    print('x_train.shape, x_test.shape, y_train.shape, y_test.shape, y_train_one.shape, y_test_one.shape')
#    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, y_train_one.shape, y_test_one.shape)
    print('x_train2.shape, x_test2.shape, y_train.shape, y_test.shape, y_train_one.shape, y_test_one.shape')
    print(x_train2.shape, x_test2.shape, y_train.shape, y_test.shape, y_train_one.shape, y_test_one.shape)
#    print('input data(x)', x_train[:5,:])
    print('input data(x)', x_train2[:5,:])
    print('answer data(y)', y_train[:5])
    print('answer data (after One Hot Vectorization)', y_train_one[:5,:])

#    x, yt = x_train, y_train_one
    x, yt, x_test = x_train2, y_train_one, x_test2

    def softmax(x):
        x = x.T
        x_max = x.max(axis=0)
        x = x - x_max
        w = np.exp(x)
        return (w / w.sum(axis=0)).T

    def pred(x, W):
        return softmax(x @ W)

    def cross_entropy(yt, yp):
        return -np.mean(np.sum(yt * np.log(yp), axis=1))

    def evaluate(x_test, y_test, y_test_one, W):
        yp_test_one = pred(x_test, W)
        yp_test = np.argmax(yp_test_one, axis=1)
        loss = cross_entropy(y_test_one, yp_test_one)
        score = accuracy_score(y_test, yp_test)
        return loss, score

    # initialize
    M = x.shape[0]
    D = x.shape[1]
    N = yt.shape[1]
    iters = 10000
    alpha = 0.01
    W = np.ones((D,N))
    print('W.shape', W.shape)
    history = np.zeros((0,3))

    # learning
    for k in range(iters):
        yp = pred(x, W)
        yd = yp - yt
        W = W - alpha * (x.T @ yd) / M

        if (k % 10 == 0):
            loss, score = evaluate(x_test, y_test, y_test_one, W)
            history = np.vstack((history, np.array([k, loss, score])))
            print('epoch = %d loss = %f score = %f' % (k, loss, score))

    print('initial state: loss %f accuracy %f' % (history[0,1], history[0,2]))
    print('final state: loss %f accuracy %f' % (history[-1,1], history[-1,2]))

    # loss function
    plt.plot(history[:,0], history[:,1])
    plt.grid()
    plt.ylim(0,1.2)
    plt.xlabel('iter', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.title('iter vs loss', fontsize=14)
    plt.show()

    # accuracy
    plt.plot(history[:,0], history[:,2])
    plt.ylim(0,1)
    plt.grid()
    plt.xlabel('iter', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    plt.title('iter vs accuracy', fontsize=14)
    plt.show()

    # 3D graph
#    x1 = np.linspace(4, 8.5, 100)
#    x2 = np.linspace(0.5, 7.5, 100)
#    xx1, xx2 = np.meshgrid(x1, x2)
#    xxx = np.array([np.ones(xx1.ravel().shape), xx1.ravel(), xx2.ravel()]).T
#    pp = pred(xxx, W)
#    c0 = pp[:,0].reshape(xx1.shape)
#    c1 = pp[:,1].reshape(xx1.shape)
#    c2 = pp[:,2].reshape(xx1.shape)
#    plt.figure(figsize=(8,8))
#    ax = plt.subplot(1,1,1,projection='3d')
#    ax.plot_surface(xx1, xx2, c0, color='lightblue', edgecolor='black', rstride=10, cstride=10, alpha=0.7)
#    ax.plot_surface(xx1, xx2, c1, color='blue', edgecolor='black', rstride=10, cstride=10, alpha=0.7)
#    ax.plot_surface(xx1, xx2, c2, color='lightgrey', edgecolor='black', rstride=10, cstride=10, alpha=0.7)
#    ax.scatter(x_t0[:,0], x_t0[:,1], 1, s=50, alpha=1, marker='+', c='k')
#    ax.scatter(x_t1[:,0], x_t1[:,1], 1, s=30, alpha=1, marker='o', c='k')
#    ax.scatter(x_t2[:,0], x_t2[:,1], 1, s=50, alpha=1, marker='x', c='k')
#    ax.set_xlim(4,8.5)
#    ax.set_ylim(0.5,7.5)
#    ax.view_init(elev=40, azim=70)
#
#    # evaluation
#    yp_test_one = pred(x_test, W)
#    yp_test = np.argmax(yp_test_one, axis=1)
#
#    score = accuracy_score(y_test, yp_test)
#    print('accuracy:%f' % score)
#    print(confusion_matrix(y_test, yp_test))
#    print(classification_report(y_test, yp_test))



