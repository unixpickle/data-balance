"""
A pixel RNN density model.
"""

import os

import numpy as np
import tensorflow as tf


def rnn_log_probs_tf(inputs):
    """
    Get the log probability of the input images.

    Args:
      inputs: a bool [batch x 28 x 28 x 1] Tensor.

    Returns:
      A batch of log probabilities.
    """
    seqs = tf.reshape(inputs, [-1, 28, 28])
    shifted = tf.concat([tf.zeros_like(seqs[:, :1]), seqs[:, :-1]], axis=1)
    rnn = _make_rnn()
    logits = tf.nn.dynamic_rnn(rnn, tf.cast(shifted, tf.float32), dtype=tf.float32)[0]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=seqs, logits=logits)
    return tf.reduce_sum(loss, axis=[1, 2])


def rnn_sample(batch_size):
    """
    Create a batch of samples from the model.

    Returns:
      A [batch x 28 x 28 x 1] Tensor.
    """
    rnn = _make_rnn()
    inputs = tf.zeros([batch_size, 28], dtype=tf.float32)
    states = rnn.zero_state(batch_size, tf.float32)
    result = []
    for _ in range(28):
        inputs, states = rnn(inputs, states)
        result.append(tf.distributions.Bernoulli(logits=inputs).sample)
    return tf.reshape(tf.stack(result, axis=1), [-1, 28, 28, 1])


def rnn_log_probs_np(inputs, checkpoint='pixel_rnn_checkpoint', batch=64):
    """
    Like rnn_log_probs_tf, but takes a numpy array and
    produces a numpy array.
    """
    with tf.Graph().as_default():
        image_ph = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        with tf.variable_scope('pixel_rnn'):
            log_probs = rnn_log_probs_tf(image_ph)
        saver = tf.train.Saver()
        all_probs = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_name(checkpoint))
            for i in range(0, len(images), batch):
                if i + batch > len(images):
                    batch_images = images[i:]
                else:
                    batch_images = images[i: i + batch]
                all_probs.extend(sess.run(log_probs, feed_dict={image_ph: batch_images}))
    return np.array(all_probs)


def checkpoint_name(dir_name):
    return os.path.join(dir_name, 'model.ckpt')


def _make_rnn():
    return tf.nn.rnn_cell.LSTMCell(num_units=128, use_peepholes=True)
