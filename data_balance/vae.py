"""
A variational autoencoder for the MNIST training set.
"""

import os

import numpy as np
import tensorflow as tf

LATENT_SIZE = 8
USE_BINARY = True


def encoder(inputs):
    """
    Encode the input images as latent vectors.

    Args:
      inputs: the input image batch.

    Returns:
      A distribution over latent vectors.
    """
    out = tf.layers.flatten(inputs)
    out = tf.layers.dense(out, 400, activation=tf.nn.relu)
    mean = tf.layers.dense(out, LATENT_SIZE)
    logstd = tf.layers.dense(out, LATENT_SIZE)
    return tf.distributions.Normal(loc=mean, scale=tf.exp(logstd))


def encoder_kl_loss(latent_dist):
    """
    Compute the encoder's mean KL loss term.

    Args:
      latent_dist: a distribution over latent codes.
    """
    zeros = tf.zeros_like(latent_dist.mean())
    batch_size = tf.cast(tf.shape(zeros)[0], zeros.dtype)
    prior = tf.distributions.Normal(loc=zeros, scale=(zeros + 1))
    total_kl = tf.reduce_sum(tf.distributions.kl_divergence(latent_dist, prior))
    mean_kl = total_kl / batch_size
    return mean_kl


def decoder(latent):
    """
    Decode the input images from latent vectors.

    Args:
      latent: the latent vector batch.

    Returns:
      A distribution over images.
    """
    out = tf.layers.dense(latent, 400, activation=tf.nn.relu)
    if USE_BINARY:
        out = tf.layers.dense(out, 28 * 28)
        out = tf.reshape(out, [-1, 28, 28, 1])
        return tf.distributions.Bernoulli(logits=out)
    else:
        mean = tf.reshape(tf.layers.dense(out, 28 * 28), [-1, 28, 28, 1])
        logstd = tf.reshape(tf.layers.dense(out, 28 * 28), [-1, 28, 28, 1])
        return tf.distributions.Normal(loc=mean, scale=tf.exp(logstd))


def vae_features(images, checkpoint='vae_checkpoint', batch=200):
    """
    Turn the batch of images into a batch of VAE features.

    Temporarily creates a graph and loads a VAE from the
    supplied checkpoint directory.

    Returns:
      A tuple (means, stddevs)
    """
    with tf.Graph().as_default():
        image_ph = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        with tf.variable_scope('encoder'):
            encoded = encoder(image_ph)
            means, stds = (encoded.loc, encoded.scale)
        saver = tf.train.Saver()
        all_means = []
        all_stds = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_name(checkpoint))
            for i in range(0, len(images), batch):
                if i + batch > len(images):
                    batch_images = images[i:]
                else:
                    batch_images = images[i: i + batch]
                sub_means, sub_stds = sess.run([means, stds], feed_dict={image_ph: batch_images})
                all_means.extend(sub_means)
                all_stds.extend(sub_stds)
    return np.array(all_means), np.array(all_stds)


def checkpoint_name(dir_name):
    return os.path.join(dir_name, 'model.ckpt')
