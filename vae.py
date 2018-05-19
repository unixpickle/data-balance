"""
A variational autoencoder for the MNIST training set.

Usage:

    $ python vae.py train --steps 5000

"""

import argparse
import itertools
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    parser = arg_parser()
    args = parser.parse_args()
    if args.command_name == 'train':
        cmd_train(args)
    elif args.command_name == 'sample':
        cmd_sample(args)
    else:
        parser.error('missing sub-command')


def cmd_train(args):
    """
    Train a VAE on the dataset.
    """
    print('Creating dataset...')
    images = input_data.read_data_sets('MNIST_data', one_hot=True).train.images
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(images)).batch(args.batch).repeat()
    batch = dataset.make_one_shot_iterator().get_next()
    batch = tf.reshape(batch, [-1, 28, 28, 1])
    print('Creating encoder...')
    with tf.variable_scope('encoder'):
        encoded = encoder(batch)
    print('Creating decoder...')
    with tf.variable_scope('decoder'):
        decoded = decoder(encoded.sample())
    print('Creating loss...')
    bool_image = tf.cast(tf.round(batch), tf.bool)
    loss = encoder_kl_loss(encoded) - tf.reduce_sum(decoded.log_prob(bool_image)) / args.batch
    print('Creating optimizer...')
    minimize = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)

    cur_step = tf.Variable(initial_value=tf.constant(0), name='global_step', trainable=False)
    inc_step = tf.assign_add(cur_step, tf.constant(1))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('Initializing variables...')
        sess.run(tf.global_variables_initializer())
        if os.path.exists(args.checkpoint):
            print('Restoring from checkpoint...')
            saver.restore(sess, checkpoint_name(args.checkpoint))
        print('Training...')
        while True:
            cur_loss, cur_step, _ = sess.run([loss, inc_step, minimize])
            print('step %d: loss=%f' % (cur_step, cur_loss))
            if cur_step % 100 == 0 or cur_step >= args.steps:
                if not os.path.exists(args.checkpoint):
                    os.mkdir(args.checkpoint)
                saver.save(sess, checkpoint_name(args.checkpoint))
            if cur_step >= args.steps:
                break


def cmd_sample(args):
    """
    Sample images from a trained VAE.
    """
    if not os.path.exists(args.checkpoint):
        sys.stderr.write('Checkpoint not found: ' + args.checkpoint + '\n')
        sys.exit(1)
    latents = tf.distributions.Normal().sample(sample_shape=[args.size ** 2, 128])
    with tf.variable_scope('decoder'):
        decoded = decoder(latents)
    images = tf.cast(decoded.mode(), tf.uint8) * tf.constant(255, dtype=tf.uint8)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('Initializing variables...')
        sess.run(tf.global_variables_initializer())
        print('Restoring parameters...')
        saver.restore(sess, checkpoint_name(args.checkpoint))
        print('Producing images...')
        images = sess.run(images)
    print('Saving output file...')
    image = np.array((args.size * 28, args.size * 28, 3))
    for i in range(args.size):
        for j in range(args.size):
            image[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28, :] = images[i * args.size + j]
    Image.fromarray(image).save(args.output)


def encoder(inputs):
    """
    Encode the input images as latent vectors.

    Args:
      inputs: the input image batch.

    Returns:
      A distribution over latent vectors.
    """
    kwargs = {'kernel_size': 3, 'strides': 2, 'padding': 'same', 'activation': tf.nn.relu}
    out = tf.layers.conv2d(inputs, filters=16, **kwargs)
    out = tf.layers.conv2d(out, filters=32, **kwargs)
    out = tf.layers.conv2d(out, filters=64, **kwargs)
    out = tf.layers.flatten(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)
    out = tf.layers.dense(out, 128, activation=tf.nn.relu)
    mean = tf.layers.dense(out, 128)
    logstd = tf.layers.dense(out, 128)
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
    kwargs = {'kernel_size': 3, 'strides': 2, 'padding': 'same', 'activation': tf.nn.relu}
    out = tf.layers.dense(latent, 256, activation=tf.nn.relu)
    out = tf.layers.dense(out, 7 * 7 * 16, activation=tf.nn.relu)
    out = tf.reshape(out, [-1, 7, 7, 16])
    out = tf.layers.conv2d(out, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
    out = tf.layers.conv2d_transpose(out, filters=32, **kwargs)
    out = tf.layers.conv2d_transpose(out, filters=32, **kwargs)
    out = tf.layers.conv2d(out, filters=3, kernel_size=3, padding='same')
    return tf.distributions.Bernoulli(logits=out)


def checkpoint_name(dir_name):
    return os.path.join(dir_name, 'model.ckpt')


def arg_parser():
    """
    Create a command-line argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='checkpoint path', default='./vae_checkpoint')

    subparsers = parser.add_subparsers(dest='command_name')

    cmd = subparsers.add_parser('train')
    cmd.add_argument('--lr', help='learning rate', default=0.001, type=float)
    cmd.add_argument('--batch', help='batch size', default=200, type=int)
    cmd.add_argument('--steps', help='total timesteps to take', default=10000, type=int)

    cmd = subparsers.add_parser('sample')
    cmd.add_argument('--size', help='sample grid side-length', default=4, type=int)
    cmd.add_argument('--output', help='output filename', default='output.png')

    return parser


if __name__ == '__main__':
    main()