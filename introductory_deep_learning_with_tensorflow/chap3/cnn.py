import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    mnist = input_data.read_data_sets("data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    # (batchsize, height, width, channel)
    img = tf.reshape(x, [-1,28,28,1])

    # convolutional layer 1
    f1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
    conv1 = tf.nn.conv2d(img, f1, strides=[1,1,1,1], padding='SAME')
    b1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.relu(conv1+b1)

    # pooling layer 1
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # convolutional layer 2
    f2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
    conv2 = tf.nn.conv2d(h_pool1, f2, strides=[1,1,1,1], padding='SAME')
    b2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(conv2 + b2)

    # pooling layer 2
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # converting to flat shape
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    # full connected layer
    w_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # output layer
    w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    out = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

    # cross entropy loss function
    y = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out + 1e-5), axis=[1]))

    # training
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)

    # evaluation
    correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt_state = tf.train.get_checkpoint_state('ckpt/')
        if ckpt_state:
            last_model = ckpt_state.model_ckeckpoint_path
            saver.restore(sess, last_model)
            print('model was loaded:', last_model)
        else:
            sess.run(init)
            print('initialized')

        test_images = mnist.test.images
        test_labels = mnist.test.labels

        last_step = sess.run(global_step)
        for i in range(1000):
            step = last_step + i
            train_images, train_labels = mnist.train.next_batch(50)
            sess.run(train_step, feed_dict={x:train_images, y:train_labels})

            if step % 100 == 0:
                acc_val = sess.run(accuracy, feed_dict={x:test_images, y:test_labels})
                print('Step %d: accuracy = %.2f' % (step + 1, acc_val))
                saver.save(sess, 'ckpt/my_model', global_step=step + 1, write_meta_graph=False)

        # saving model
        #saver.save(sess, 'ckpt/my_model')

