import math
import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MB_SIZE = 64
STEPS_N = 28
INPUT_N = 784 // STEPS_N
CLASS_N = 10

REPORT_INTERVAL = 1000
TEST_INTERVAL = 10000

TEST_N = 5000


def build_model():
    x = tf.placeholder(tf.float32, [None, STEPS_N, INPUT_N])
    y_true = tf.placeholder(tf.float32, [None, CLASS_N])
    w = tf.Variable(tf.truncated_normal(shape=[STEPS_N * INPUT_N, CLASS_N]), name='weights')
    b = tf.Variable(tf.constant(0., shape=[CLASS_N]), name='bias')

    signal = tf.reshape(x, [-1, STEPS_N * INPUT_N])

    y_pred = tf.matmul(signal, w) + b
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


def test(sess, model, mnist):
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
    print("test loss", numpy.average(losses), "accuracy", numpy.average(accuracies))


def train():
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
                print("train loss", numpy.average(train_losses[-REPORT_INTERVAL:]),
                      "accuracy", numpy.average(train_accuracies[-REPORT_INTERVAL:]))

            if (batch_idx + 1) % TEST_INTERVAL == 0:
                test(sess, model, mnist)


if __name__ == '__main__':
    train()
