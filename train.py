import contextlib
import os
import math

import datetime
import random

import numpy
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.saver import Saver

LOG_DIR = "../lstm-data/logs/"
CHECKPOINT_DIR = "../lstm-data/checkpoints/"
CHECKPOINT_FILE_NAME = os.path.join(CHECKPOINT_DIR, "lstm")

MB_SIZE = 128
STEPS_N = 28
INPUT_N = 784 // STEPS_N
CLASS_N = 10

REPORT_INTERVAL = 1000
TEST_INTERVAL = 10000

TEST_N = 5000


def assert_tensor_shape(tensor: tf.Tensor, expected):
    shape = tuple(tensor.get_shape().as_list())
    _assert_shapes_equal(expected, shape)


def assert_array_shape(array: numpy.array, expected):
    _assert_shapes_equal(array.shape, expected)


def _assert_shapes_equal(expected, shape):
    error_message = "shapes do not match {} vs {}".format(shape, expected)
    assert len(shape) == len(expected), error_message
    for actual_val, expected_val in zip(shape, expected):
        assert expected_val is not None or expected_val == actual_val, error_message


class Logger:
    def __init__(self, save_dir, prefix):
        self.save_dir = save_dir
        self.prefix = prefix

        self.handle = None

        with contextlib.suppress(FileExistsError):
            os.makedirs(save_dir)

    def log(self, *args, **kwargs):
        print(datetime.datetime.now().time(), *args, **kwargs, flush=True)
        print(datetime.datetime.now().time(), *args, **kwargs, file=self.handle, flush=True)

    def open(self):
        self.handle = open(os.path.join(self.save_dir, "{}-{}".format(
            self.prefix,
            datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))), "x")
        return self.handle


def lstm_layer(hidden_n, steps_n, input_n, x):
    forget_bias = 1.
    initial_bias = numpy.zeros([4 * hidden_n], dtype=numpy.float32)
    initial_bias[1 * hidden_n: 2 * hidden_n] += forget_bias

    lstm_w = tf.Variable(tf.truncated_normal(shape=[input_n + hidden_n, hidden_n * 4]), name='lstm_weights')
    lstm_b = tf.Variable(initial_value=initial_bias, name='lstm_bias')

    c_0 = tf.Variable(tf.zeros(shape=[1, hidden_n]), name='c_0')
    h_0 = tf.Variable(tf.zeros(shape=[1, hidden_n]), name='h_0')

    c = [tf.tile(c_0, [tf.shape(x)[0], 1])]
    h = [tf.tile(h_0, [tf.shape(x)[0], 1])]

    for t in range(steps_n):
        x_t = x[:, t, :]
        input_ = tf.concat([x_t, h[t]], axis=1)

        z = tf.matmul(input_, lstm_w) + lstm_b
        i = tf.sigmoid(z[:, 0 * hidden_n:1 * hidden_n])
        f = tf.sigmoid(z[:, 1 * hidden_n:2 * hidden_n])
        o = tf.sigmoid(z[:, 2 * hidden_n:3 * hidden_n])
        g = tf.tanh(z[:, 3 * hidden_n:4 * hidden_n])

        c.append(f * c[t] + i * g)
        h.append(o * tf.tanh(c[t + 1]))

    assert_tensor_shape(h[-1], [None, hidden_n])

    return {
        "signal": h[-1],
        "c": tf.transpose(tf.stack(c), perm=[1, 0, 2]),
        "h": tf.transpose(tf.stack(h), perm=[1, 0, 2])
    }


def build_model():
    x = tf.placeholder(tf.float32, [None, STEPS_N, INPUT_N])
    y_true = tf.placeholder(tf.float32, [None, CLASS_N])

    hidden_n = 28
    result = lstm_layer(hidden_n, STEPS_N, INPUT_N, x)
    signal = lstm_layer(CLASS_N, STEPS_N, hidden_n, result['h'])['signal']
    assert_tensor_shape(signal, (None, 10))
    y_pred = signal

    # fc_w = tf.Variable(tf.truncated_normal(shape=[hidden_n, CLASS_N]), name='fc_weights')
    # fc_b = tf.Variable(tf.zeros(shape=[CLASS_N]), name='fc_bias')
    # y_pred = tf.matmul(signal, fc_w) + fc_b

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1)), tf.float32))
    step = tf.train.AdamOptimizer().minimize(loss)
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


def _cut_begin_end(orig, new):
    assert orig >= new
    diff = orig - new
    begin = int(math.floor(diff / 2))
    bottom_clip = int(math.ceil(diff / 2))
    end = orig - bottom_clip
    assert begin <= end
    assert begin + 1 >= bottom_clip
    assert begin + new + bottom_clip == orig
    return begin, end


def crop_image(x, w, h):
    x = numpy.array(x)
    assert len(x.shape) == 2
    orig_w, orig_h = x.shape
    assert orig_w >= w
    assert orig_h >= h

    cut_w = _cut_begin_end(orig_w, w)
    cut_h = _cut_begin_end(orig_h, h)

    return x[cut_w[0]:cut_w[1], cut_h[0]:cut_h[1]]


def pad(array, to_length):
    array = numpy.array(array)
    assert len(array.shape) == 1
    assert array.shape[0] <= to_length
    return numpy.pad(array, (0, to_length - array.shape[0]), constant_values=0., mode='constant')


def augment(batch_x: numpy.array):
    num = batch_x.shape[0]
    batch_x = batch_x.reshape((-1, 28, 28))
    widths = [random.randint(24, 28) for _ in range(num)]
    heights = [random.randint(24, 28) for _ in range(num)]

    tab = []
    for x, w, h in zip(batch_x, widths, heights):
        item = crop_image(x, w, h)
        item = item.reshape(-1)
        item = pad(item, 28 * 28)
        tab.append(item)

    tab = numpy.array(tab)
    assert_array_shape(tab, [None, 28 * 28])

    return tab


def train(log):
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    model = build_model()
    saver = Saver()

    train_losses = []
    train_accuracies = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for batch_idx in range(1000000):
            batch_x, batch_y = mnist.train.next_batch(MB_SIZE)
            batch_x = transform_batch(batch_x)
            # batch_x = augment(batch_x)
            # batch_x = batch_x.reshape((-1, 28, 28))

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
                with contextlib.suppress(FileExistsError):
                    os.makedirs(CHECKPOINT_DIR)
                saver.save(sess, CHECKPOINT_FILE_NAME)


def read_image(name):
    image = Image.open(name)
    return numpy.asarray(image)


def predict(file_name):
    model = build_model()
    saver = Saver()

    x = read_image(file_name)
    x = x.reshape((1, 28, 28))
    x = x.astype(numpy.float32) / 255.

    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_FILE_NAME)

        y_pred = sess.run([model['y_pred']], feed_dict={
            model['x']: x
        })

        print(y_pred)


def main():
    logger = Logger(LOG_DIR, "train")
    with logger.open():
        train(logger.log)


if __name__ == '__main__':
    main()
