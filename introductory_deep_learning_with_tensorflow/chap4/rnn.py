import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    # rnn which has 100 middle layer units
    cell_1 = tf.nn.rnn_cell.BasicRNNCell(num_units=100)

    # lstm
    cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=100, use_peepholes=True)

    # dropout
    cell_2 = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.6)

    # multi layer rnn
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell_1, cell_2])

    # dynamic_rnn
    max_time = 50
    input_size = 10

    x = tf.placeholder(tf.float32, [None, max_time, input_size])
    cell = tf.nn.rnn_cell.LSTMCell(num_units=100, use_peepholes=True)
    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32)

    last_output = outputs[:, -1, :]
    w = tf.Variable(tf.truncated_normal([128, 10], stddev=0.1))
    b = tf.Variable(tf.zeros([10]))

    out = tf.nn.softmax(tf.matmul(last_output, w) + b)

    # static_rnn
#    inputs = []
#    for i in range(time_step):
#        x = tf.placeholder(tf.float32, [None, input_size])
#        inputs.append(x)
#        tf.add_to_collection("x", x)

