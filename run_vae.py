"""
Train, sample, and evaluate a VAE for MNIST data.

Usage:

    $ python run_vae.py train
    $ python run_vae.py sample --output samples.png

"""

import argparse
import os
import sys

from PIL import Image
import numpy as np
import tensorflow as tf

from data_balance.data import balancing_task, mnist_training_batch
from data_balance.vae import LATENT_SIZE, checkpoint_name, decoder, encoder, encoder_kl_loss, vae_features


def main():
    parser = arg_parser()
    args = parser.parse_args()
    cmds = {
        'train': cmd_train,
        'sample': cmd_sample,
        'balance': cmd_balance,
        'reconstruct': cmd_reconstruct,
    }
    if args.command_name in cmds:
        cmds[args.command_name](args)
    else:
        parser.error('missing sub-command')


def cmd_train(args):
    """
    Train a VAE on the dataset.
    """
    print('Creating dataset...')
    images = mnist_training_batch(args.batch)
    print('Creating encoder...')
    with tf.variable_scope('encoder'):
        encoded = encoder(images)
    print('Creating decoder...')
    with tf.variable_scope('decoder'):
        decoded = decoder(encoded.sample())
    print('Creating loss...')
    bool_images = tf.cast(tf.round(images), tf.bool)
    loss = encoder_kl_loss(encoded) - tf.reduce_sum(decoded.log_prob(bool_images)) / args.batch
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
            if cur_step % args.save_interval == 0 or cur_step >= args.steps:
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
    latent_prior = tf.distributions.Normal(loc=0.0, scale=1.0)
    latents = latent_prior.sample(sample_shape=[args.size ** 2, LATENT_SIZE])
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
    image = np.zeros((args.size * 28, args.size * 28, 3), dtype='uint8')
    for i in range(args.size):
        for j in range(args.size):
            image[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28, :] = images[i * args.size + j]
    Image.fromarray(image).save(args.output)


def cmd_balance(args):
    """
    Compute how class-balanced the means of the latent
    codes are.
    """
    images, labels = balancing_task(list(range(10)), [1.0] * 10)
    features, _ = vae_features(images, checkpoint=args.checkpoint)
    with tf.Session() as sess:
        dist = tf.distributions.Normal(loc=0.0, scale=1.0)
        logits = tf.reduce_sum(dist.log_prob(features), axis=-1)
        probs = sess.run(tf.nn.softmax(logits))
    for class_idx in range(10):
        print('class %d: %f' % (class_idx, np.sum(probs[labels == class_idx])))


def cmd_reconstruct(args):
    """
    Generate a picture of reconstructions.
    """
    if not os.path.exists(args.checkpoint):
        sys.stderr.write('Checkpoint not found: ' + args.checkpoint + '\n')
        sys.exit(1)
    images = mnist_training_batch(args.size ** 2, validation=True)
    with tf.variable_scope('encoder'):
        encoded = encoder(images).mean()
    with tf.variable_scope('decoder'):
        decoded = tf.cast(decoder(encoded).mode(), tf.float32)
    images = tf.concat([tf.tile(images, [1, 1, 1, 3]), decoded], axis=0)
    images = tf.cast(images * 255, tf.uint8)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('Initializing variables...')
        sess.run(tf.global_variables_initializer())
        print('Restoring parameters...')
        saver.restore(sess, checkpoint_name(args.checkpoint))
        print('Producing images...')
        images = sess.run(images)
    print('Saving output file...')
    image = np.zeros((args.size * 28 * 2, args.size * 28, 3), dtype='uint8')
    for i in range(args.size * 2):
        for j in range(args.size):
            image[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28, :] = images[i * args.size + j]
    Image.fromarray(image).save(args.output)


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
    cmd.add_argument('--save-interval', help='steps per save', default=500, type=int)

    cmd = subparsers.add_parser('sample')
    cmd.add_argument('--size', help='sample grid side-length', default=4, type=int)
    cmd.add_argument('--output', help='output filename', default='output.png')

    cmd = subparsers.add_parser('balance')

    cmd = subparsers.add_parser('reconstruct')
    cmd.add_argument('--size', help='sample grid side-length', default=4, type=int)
    cmd.add_argument('--output', help='output filename', default='output.png')

    return parser


if __name__ == '__main__':
    main()
