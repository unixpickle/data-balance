"""
Train, sample, and evaluate a pixel RNN for MNIST.

Usage:

    $ python run_pixel_rnn.py train
    $ python run_pixel_rnn.py sample --output samples.png

"""

import argparse
import os
import sys

from PIL import Image
import numpy as np
import tensorflow as tf

from data_balance.data import balancing_task, mnist_training_batch
from data_balance.pixel_rnn import checkpoint_name, rnn_log_probs_np, rnn_log_probs_tf, rnn_sample


def main():
    parser = arg_parser()
    args = parser.parse_args()
    cmds = {
        'train': cmd_train,
        'sample': cmd_sample,
        'balance': cmd_balance,
    }
    if args.command_name in cmds:
        cmds[args.command_name](args)
    else:
        parser.error('missing sub-command')


def cmd_train(args):
    """
    Train an RNN on the dataset.
    """
    print('Creating dataset...')
    images = mnist_training_batch(args.batch) > 0.5
    print('Creating RNN...')
    with tf.variable_scope('pixel_rnn'):
        log_probs = rnn_log_probs_tf(images)
    print('Creating loss...')
    loss = tf.reduce_mean(log_probs)
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
    Sample images from a trained RNN.
    """
    if not os.path.exists(args.checkpoint):
        sys.stderr.write('Checkpoint not found: ' + args.checkpoint + '\n')
        sys.exit(1)
    with tf.variable_scope('pixel_rnn'):
        images = rnn_sample(args.size ** 2)
    images = tf.cast(images, tf.uint8) * 255
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
    log_probs = rnn_log_probs_np(images > 0.5, checkpoint=args.checkpoint)
    log_probs = np.exp(log_probs - np.max(log_probs))
    total = np.sum(log_probs)
    for class_idx in range(10):
        print('class %d: %f' % (class_idx, np.sum(log_probs[labels == class_idx]) / total))


def arg_parser():
    """
    Create a command-line argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='checkpoint path', default='./pixel_rnn_checkpoint')

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

    return parser


if __name__ == '__main__':
    main()
