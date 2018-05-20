"""
Class balancers.
"""

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
from sklearn.mixture import GaussianMixture
import tensorflow as tf

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
