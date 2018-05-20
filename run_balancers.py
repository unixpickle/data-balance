"""
Evaluate all the balancers.
"""

import numpy as np

from data_balance.balance import DensityBalancer, UniformBalancer, VoronoiBalancer, ClusterBalancer
from data_balance.data import balancing_task

VAE_CHECKPOINT = 'vae_checkpoint'


def main():
    print('Creating tasks...')
    tasks = {
        '2, 3 (balanced)': balancing_task([2, 3], [1, 1]),
        '2 (x2), 3 (x1)': balancing_task([2, 3], [1, 1], dups=[2, 1]),
        '5 (10%), 1 (90%)': balancing_task([5, 1], [0.1, 0.9]),
        '2 (10%), 3 (90%)': balancing_task([2, 3], [0.1, 0.9]),
        '3 (10%), 2 (90%)': balancing_task([3, 2], [0.1, 0.9]),
    }
    print('Creating balancers...')
    balancers = {
        'uniform': UniformBalancer(),
        'density': DensityBalancer(VAE_CHECKPOINT),
        'voronoi': VoronoiBalancer(VAE_CHECKPOINT),
        'box_voronoi': VoronoiBalancer(VAE_CHECKPOINT, use_box=True),
        'smooth_voronoi': VoronoiBalancer(VAE_CHECKPOINT, use_box=True, smooth=0.01),
        'cluster': ClusterBalancer(VAE_CHECKPOINT)
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
