"""
Class balancers.
"""

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import tensorflow as tf

from .data import images_training_batch, mnist_training_batch, read_mnist
from .vae import vae_features


class Balancer(ABC):
    """
    An algorithm that can balance a dataset.
    """
    @abstractmethod
    def assign_weights(self, images):
        """
        For a batch of images, produce a batch of weights.

        The goal is that the sum of the weights for a
        given class is the same as the sum of the weights
        for every other represented class.
        """
        pass


class UniformBalancer(Balancer):
    """
    A balancer that returns uniform weights.
    """

    def assign_weights(self, images):
        return np.zeros([len(images)], dtype='float32') + 1


class VAEBalancer(Balancer):
    """
    A Balancer that operates on VAE features.
    """

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def assign_weights(self, images):
        means, stds = vae_features(images, checkpoint=self.checkpoint)
        return self.vae_weights(means, stds)

    @abstractmethod
    def vae_weights(self, means, stds):
        """
        Assign weights based on VAE features.
        """
        pass


class VoronoiBalancer(VAEBalancer):
    """
    A balancer that weights samples based on the densities
    within their Voronoi cells.
    """

    def __init__(self, checkpoint, use_box=False, smooth=0.0):
        super(VoronoiBalancer, self).__init__(checkpoint)
        self._use_box = use_box
        self._smooth = smooth

    def vae_weights(self, features, _):
        counts = np.zeros([len(features)], dtype='float32')
        noises = self._noise_samples(features)
        for noise in noises:
            sq_dists = np.sum(np.square(features - noise), axis=-1)
            if self._smooth > 0:
                cutoff = np.percentile(sq_dists, (1 - self._smooth) * 100)
                counts[sq_dists > cutoff] += 1
            else:
                neighbor_idx = np.argmin(sq_dists)
                counts[neighbor_idx] += 1
        return counts

    def _noise_samples(self, features):
        """
        Filter the noise vectors to those that should
        actually be used for sampling.
        """
        latent_size = len(features[-1])
        num_samples = len(features) * self._samples_per_image()
        if not self._use_box:
            return np.random.normal(size=[num_samples, latent_size])
        min_coords = np.min(features, axis=0)
        max_coords = np.max(features, axis=0)
        return (np.random.uniform(size=[num_samples, latent_size]) *
                (max_coords - min_coords)) + min_coords

    @staticmethod
    def _samples_per_image():
        """
        Get the ratio of sampled noise vectors to input
        images.
        """
        return 1


class ClusterBalancer(VAEBalancer):
    """
    A balancer that uses a clustering algorithm.
    """

    def __init__(self, checkpoint, num_clusters=10):
        super(ClusterBalancer, self).__init__(checkpoint)
        self.num_clusters = num_clusters

    def vae_weights(self, features, _):
        mixture = GaussianMixture(n_components=self.num_clusters)
        mixture.fit(features)
        classes = mixture.predict(features)
        counts = Counter(classes)
        return np.array([1 / counts[label] for label in classes])


class KDEBalancer(VAEBalancer):
    """
    A balancer that uses a kernel density estimator.
    """

    def vae_weights(self, means, stds):
        mixture = KernelDensity()
        mixture.fit(means)

        with tf.Graph().as_default():
            log_probs = tf.reduce_sum(tf.distributions.Normal(loc=0.0, scale=1.0).log_prob(means),
                                      axis=-1)
            with tf.Session() as sess:
                log_probs = sess.run(log_probs)
        logits = log_probs - mixture.score_samples(means)
        logits -= np.max(logits)
        return np.exp(logits)


class DensityBalancer(VAEBalancer):
    """
    A balancer that weights each datapoint according to
    how well the datapoint can be identified by its mean.
    """

    def __init__(self, checkpoint, temperature=0.01, num_samples=1):
        super(DensityBalancer, self).__init__(checkpoint)
        self._temperature = temperature
        self._num_samples = num_samples

    def vae_weights(self, means, stds):
        with tf.Graph().as_default():
            mean_const = tf.constant(means)
            data_dist = tf.distributions.Normal(loc=mean_const, scale=stds)

            def loop_body(t, result_arr):
                point_samples = tf.expand_dims(mean_const[t], 0)
                point_samples = tf.tile(point_samples, [len(means), 1])
                logits = tf.reduce_sum(data_dist.log_prob(point_samples), axis=-1)
                ownership = tf.nn.softmax(logits * self._temperature)
                return t + 1, result_arr.write(t, ownership[t])

            arr = tf.TensorArray(tf.float32, size=len(means))
            _, final_weights = tf.while_loop(cond=lambda t, _: t < len(means),
                                             body=loop_body,
                                             loop_vars=[tf.constant(0), arr])

            with tf.Session() as sess:
                return sess.run(final_weights.stack())


class TrainBalancer(Balancer):
    """
    A balancer that tries to discriminate between the test
    set and the training set, and weights samples based on
    the log-loss under this classifier.
    """

    def __init__(self, checkpoint=None, num_iters=1000, batch_size=200, lr=0.001, debug=False):
        assert batch_size % 2 == 0
        self._checkpoint = checkpoint
        self._use_vae = checkpoint is not None
        self._num_iters = num_iters
        self._batch_size = batch_size
        self._lr = lr
        self._debug = debug

    def assign_weights(self, images):
        with tf.Graph().as_default():
            training_batch, testing_batch, eval_batch, eval_feed = self._network_inputs(images)
            batch = tf.concat([training_batch, testing_batch], axis=0)
            labels = tf.cast(tf.range(self._batch_size) >= (self._batch_size // 2), tf.float32)
            labels = tf.expand_dims(labels, axis=1)
            with tf.variable_scope('model'):
                logits = self._apply_network(batch)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
                loss = tf.reduce_mean(loss)
            with tf.variable_scope('model', reuse=True):
                eval_logits = self._apply_network(eval_batch)
            minimize = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(loss)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for _ in range(self._num_iters):
                    _, loss_val = sess.run([minimize, loss])
                    if self._debug:
                        print(loss_val)
                log_weights = sess.run(eval_logits, feed_dict=eval_feed)
        log_weights -= np.max(log_weights)
        # TODO: why should this be negative?
        return np.exp(-log_weights)

    def _network_inputs(self, images):
        if not self._use_vae:
            ph = tf.placeholder(tf.float32, shape=images.shape)
            return (mnist_training_batch(self._batch_size // 2),
                    images_training_batch(images, self._batch_size // 2),
                    ph, {ph: images})
        train_means, _ = vae_features(read_mnist().train.images.reshape([-1, 28, 28, 1]),
                                      checkpoint=self._checkpoint)
        test_means, _ = vae_features(images, checkpoint=self._checkpoint)
        ph = tf.placeholder(tf.float32, shape=test_means.shape)
        return (images_training_batch(train_means, self._batch_size // 2),
                images_training_batch(test_means, self._batch_size // 2),
                ph, {ph: test_means})

    def _apply_network(self, batch):
        if self._use_vae:
            out = tf.layers.dense(batch, 64, activation=tf.nn.leaky_relu)
            out = tf.layers.dense(batch, 32, activation=tf.nn.leaky_relu)
        else:
            out = tf.layers.conv2d(batch, 16, 3, strides=2, activation=tf.nn.leaky_relu)
            out = tf.layers.conv2d(out, 32, 3, strides=2, activation=tf.nn.leaky_relu)
            out = tf.layers.conv2d(out, 64, 3, activation=tf.nn.leaky_relu)
            out = tf.reduce_mean(out, axis=[1, 2])
        return tf.layers.dense(out, 1, kernel_initializer=tf.zeros_initializer())
