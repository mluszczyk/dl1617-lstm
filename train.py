import math

import datetime
import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MB_SIZE = 128
STEPS_N = 28
INPUT_N = 784 // STEPS_N
CLASS_N = 10

REPORT_INTERVAL = 1000
TEST_INTERVAL = 10000

TEST_N = 5000


def datetime_log(*args, **kwargs):
    print(datetime.datetime.now().time(), *args, **kwargs)


def build_model():
    x = tf.placeholder(tf.float32, [None, STEPS_N, INPUT_N])
    y_true = tf.placeholder(tf.float32, [None, CLASS_N])

    hidden_n = 28

    forget_bias = 1.
    initial_bias = numpy.zeros([4 * hidden_n], dtype=numpy.float32)
    initial_bias[1 * hidden_n: 2 * hidden_n] += forget_bias

    lstm_w = tf.Variable(tf.truncated_normal(shape=[INPUT_N + hidden_n, hidden_n * 4]), name='lstm_weights')
    lstm_b = tf.Variable(initial_value=initial_bias, name='lstm_bias')

    fc_w = tf.Variable(tf.truncated_normal(shape=[hidden_n, CLASS_N]), name='fc_weights')
    fc_b = tf.Variable(tf.zeros(shape=[CLASS_N]), name='fc_bias')

    c_0 = tf.Variable(tf.zeros(shape=[1, hidden_n]), name='c_0')
    h_0 = tf.Variable(tf.zeros(shape=[1, hidden_n]), name='h_0')

    c = [tf.tile(c_0, [tf.shape(x)[0], 1])]
    h = [tf.tile(h_0, [tf.shape(x)[0], 1])]

    for t in range(STEPS_N):
        x_t = x[:, t, :]
        input_ = tf.concat([x_t, h[t]], axis=1)

        z = tf.matmul(input_, lstm_w) + lstm_b
        i = tf.sigmoid(z[:, 0 * hidden_n:1 * hidden_n])
        f = tf.sigmoid(z[:, 1 * hidden_n:2 * hidden_n])
        o = tf.sigmoid(z[:, 2 * hidden_n:3 * hidden_n])
        g = tf.tanh(z[:, 3 * hidden_n:4 * hidden_n])

        c.append(f * c[t] + i * g)
        h.append(o * tf.tanh(c[t + 1]))

    signal = h[-1]
    y_pred = tf.matmul(signal, fc_w) + fc_b

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1)), tf.float32))
    step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return {
        "x": x,
        "y_pred": y_pred,
        "y_true": y_true,
        "loss": loss,
        "step": step,
        "accuracy": accuracy
    }


def transform_batch(batch_x):
    return batch_x.reshape((MB_SIZE, STEPS_N, INPUT_N))


def test(log, sess, model, mnist):
    losses = []
    accuracies = []
    for x in range(int(math.ceil(TEST_N / MB_SIZE))):
        batch_x, batch_y = mnist.test.next_batch(MB_SIZE)
        batch_x = transform_batch(batch_x)
        loss, accuracy = sess.run([model['loss'], model['accuracy']], feed_dict={
            model['x']: batch_x,
            model['y_true']: batch_y
        })
        losses.append(loss)
        accuracies.append(accuracy)
    log("test loss", numpy.average(losses), "accuracy", numpy.average(accuracies))


def train(log):
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    model = build_model()

    train_losses = []
    train_accuracies = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for batch_idx in range(1000000):
            batch_x, batch_y = mnist.train.next_batch(MB_SIZE)
            batch_x = transform_batch(batch_x)

            output = sess.run([model['step'], model['loss'], model['accuracy']], feed_dict={
                model['x']: batch_x,
                model['y_true']: batch_y
            })
            train_losses.append(output[1])
            train_accuracies.append(output[2])

            if (batch_idx + 1) % REPORT_INTERVAL == 0:
                log("train loss", numpy.average(train_losses[-REPORT_INTERVAL:]),
                      "accuracy", numpy.average(train_accuracies[-REPORT_INTERVAL:]))

            if (batch_idx + 1) % TEST_INTERVAL == 0:
                test(log, sess, model, mnist)


if __name__ == '__main__':
    train(datetime_log)
