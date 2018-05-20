"""
Evaluate all the balancers.
"""

import numpy as np

from data_balance.balance import UniformBalancer, VoronoiBalancer
from data_balance.data import balancing_task

VAE_CHECKPOINT = 'vae_checkpoint'


def main():
    print('Creating tasks...')
    tasks = {'10% 2, 90% 3': balancing_task([2, 3], [0.1, 0.9])}
    print('Creating balancers...')
    balancers = {
        'uniform': UniformBalancer(),
        'voronoi': VoronoiBalancer(VAE_CHECKPOINT),
        'box_voronoi': VoronoiBalancer(VAE_CHECKPOINT, use_box=True)
    }

    for task_name, (images, classes) in tasks.items():
        print('Evaluating on %s...' % task_name)
        for balancer_name, balancer in balancers.items():
            weights = balancer.assign_weights(images)
            print('%s: %s' % (balancer_name, class_weights(classes, weights)))


def class_weights(classes, weights):
    weights = weights / np.sum(weights)
    res = {}
    for class_num in sorted(set(classes)):
        res[class_num] = np.sum(weights[classes == class_num])
    return res


if __name__ == '__main__':
    main()
