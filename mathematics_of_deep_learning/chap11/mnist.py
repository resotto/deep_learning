import os
import platform
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numpy as np
from IPython.display import set_matplotlib_formats
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

if __name__ == '__main__':
    if platform.system() == 'Darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    set_matplotlib_formats('png', 'pdf')

    # approzimate calculation
    def f(x):
        return np.exp(x)

    h = 0.001
    diff = (f(0 + h) - f(0 - h)) / (2 * h)
    print('diff:', diff)

    D = 784
    H = 128
    num_classes = 10
    (x_train_org, y_train), (x_test_org, y_test) = mnist.load_data()

    x_train = x_train_org.reshape(-1, D) / 255.0
    x_test = x_test_org.reshape((-1, D)) / 255.0

    y_train_ohe = np_utils.to_categorical(y_train, num_classes)
    y_test_ohe = np_utils.to_categorical(y_test, num_classes)

    batch_size = 512
    nb_epoch = 50

    # SGD
    model = Sequential()
    model.add(Dense(H, activation='relu', kernel_initializer='he_normal', input_shape=(D,)))
    model.add(Dense(H, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])

    history1 = model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=nb_epoch, verbose=1, \
        validation_data=(x_test, y_test_ohe))

    # RmsProp
    model = Sequential()
    model.add(Dense(H, activation='relu', kernel_initializer='he_normal', input_shape=(D,)))
    model.add(Dense(H, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])

    history2 = model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=nb_epoch, verbose=1, \
        validation_data=(x_test, y_test_ohe))

    # Momentum
    model = Sequential()
    model.add(Dense(H, activation='relu', kernel_initializer='he_normal', input_shape=(D,)))
    model.add(Dense(H, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))

    sgd = optimizers.SGD(momentum = 0.9)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])

    history3 = model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=nb_epoch, verbose=1, \
        validation_data=(x_test, y_test_ohe))

    # loss function
    plt.figure(figsize=(8,6))
    plt.plot(history1.history['val_loss'], label='SGD', lw=3, c='k')
    plt.plot(history2.history['val_loss'], label='rmsprop', lw=3, c='b', linestyle='dashed')
    plt.plot(history3.history['val_loss'], label='momentum', lw=3, c='k', linestyle='dashed')
    plt.ylim(0,2)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.legend(fontsize=14)
    plt.show()

    # accuracy
    plt.figure(figsize=(8,6))
    plt.plot(history1.history['val_acc'], label='SGD', lw=3, c='k')
    plt.plot(history2.history['val_acc'], label='rmsprop', lw=3, c='b')
    plt.plot(history3.history['val_acc'], label='momentum', lw=3, c='k', linestyle='dashed')
    plt.ylim(0.8,1)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.legend(fontsize=14)
    plt.show()

    div = 8
    dim = 8
    p = [-1, 1, -3, 8, -7]
    xMin = -2
    xMax = 1

    x = np.linspace(xMin, xMax, num=div)
    xx = np.linspace(xMin, xMax, num=div*10)
    y = np.polyval(p, x)
    yy = np.polyval(p, xx)
    z = y + 5 * np.random.randn(div)

    def print_fix(x):
        [print('{:.3f}'.format(n)) for n in x]

    def print_fix_model(m):
       w = m.coef_.tolist()
       w[0] = m.intercept_
       print_fix(w)

    def f(x):
        return [x**i for i in range(dim)]

    X = [f(x0) for x0 in x]
    XX = [f(x0) for x0 in xx]

    model = LinearRegression().fit(X,z)
    yy_pred = model.predict(XX)

    model2 = Ridge(alpha = 0.5).fit(X,z)
    yy_pred2 = model2.predict(XX)

    plt.figure(figsize=(8,6))
    plt.plot(xx, yy, label='polynomial', lw=1, c='k')
    plt.scatter(x, z, label='observed', s=50, c='k')
    plt.plot(xx, yy_pred, label='linear regression', lw=3, c='k')
    plt.plot(xx, yy_pred2, label='L2 regularizer', lw=3, c='b')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.legend(fontsize=14)
    plt.show()





































































