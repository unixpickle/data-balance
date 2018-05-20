"""
Generate balancing tasks.
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def mnist_training_batch(batch_size, validation=False):
    """
    Create a Tensor that fetches batches of images from
    the MNIST dataset.

    Returns:
      A [batch_size x 28 x 28 x 1] Tensor.
    """
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    if validation:
        images = dataset.validation.images
    else:
        images = dataset.train.images
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(images)).batch(batch_size).repeat()
    batch = dataset.make_one_shot_iterator().get_next()
    return tf.reshape(batch, [-1, 28, 28, 1])


def random_balancing_task(num_classes=2, validation=True):
    """
    Create a random class balancing task.

    Args:
      num_classes: the number of classes to include.
      validation: a flag indicating if the validation set
        should be used (versus the test set).
    """
    all_classes = np.arange(10)
    np.random.shuffle(all_classes)
    classes = all_classes[:num_classes]
    amounts = np.random.uniform(size=num_classes)
    return balancing_task(classes, amounts)


def balancing_task(classes, fractions, dups=None, validation=True):
    """
    Generate a data balancing task.

    Args:
      classes: a list of classes to include, where each
        class is a number from 0 to 9.
      fractions: the fraction of each class's test data to
        use (one per class in classes).
      dups: if specified, a list specifying, for each
        class, the number of times that class's data is
        repeated.

    Returns:
      A tuple (images, labels), where images is a batch of
        28x28x1 images, and labels is a batch of integers.
    """
    if dups is None:
        dups = [1] * len(classes)
    dataset = input_data.read_data_sets('MNIST_data', one_hot=False)
    if validation:
        dataset = dataset.validation
    else:
        dataset = dataset.test
    images = []
    labels = []
    for class_idx, frac, num_dups in zip(classes, fractions, dups):
        all_images = dataset.images[dataset.labels == class_idx]
        num_images = min(len(all_images), max(0, int(frac * len(all_images))))
        np.random.shuffle(all_images)
        for _ in range(num_dups):
            images.extend(all_images[:num_images])
            labels.extend([class_idx] * num_images)
    return np.array(images).reshape([-1, 28, 28, 1]), np.array(labels, dtype='int32')
