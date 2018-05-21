"""
Evaluate all the balancers.
"""

import numpy as np

from data_balance.balance import (DensityBalancer, KDEBalancer, UniformBalancer, VoronoiBalancer,
                                  ClusterBalancer)
from data_balance.data import balancing_task

VAE_CHECKPOINT = 'vae_checkpoint'


def main():
    print('Creating tasks...')
    tasks = {
        '2, 3 (balanced)': balancing_task([2, 3], [1, 1]),
        '2 (x2), 3 (x1)': balancing_task([2, 3], [1, 1], dups=[2, 1]),
        '5 (10%), 1 (90%)': balancing_task([5, 1], [0.1, 0.9]),
        '7 (10%), 8 (90%)': balancing_task([7, 8], [0.1, 0.9]),
        '8 (10%), 7 (90%)': balancing_task([8, 7], [0.1, 0.9]),
        '2 (10%), 3 (90%)': balancing_task([2, 3], [0.1, 0.9]),
        '3 (10%), 2 (90%)': balancing_task([3, 2], [0.1, 0.9]),
        '6 (30%), 9 (70%)': balancing_task([6, 9], [0.3, 0.7]),
        '4 (30%), 9 (70%)': balancing_task([4, 9], [0.3, 0.7]),
    }
    print('Creating balancers...')
    balancers = {
        'uniform': UniformBalancer(),
        'density': DensityBalancer(VAE_CHECKPOINT),
        'voronoi': VoronoiBalancer(VAE_CHECKPOINT),
        'box_voronoi': VoronoiBalancer(VAE_CHECKPOINT, use_box=True),
        'smooth_voronoi': VoronoiBalancer(VAE_CHECKPOINT, use_box=True, smooth=0.01),
        'cluster': ClusterBalancer(VAE_CHECKPOINT),
        'kde': KDEBalancer(VAE_CHECKPOINT)
    }

    print('| Task | ' + ' | '.join(balancers.keys()) + ' |')
    print('|:-:|' + '|'.join([':-:'] * len(balancers.keys())) + '|')

    for task_name, (images, classes) in tasks.items():
        entropies = []
        for balancer in balancers.values():
            weights = balancer.assign_weights(images)
            entropies.append(class_entropy(classes, weights))
        max_idx = np.argmax(entropies)
        value_strs = []
        for i, entropy in enumerate(entropies):
            value_str = '%.3f' % entropy
            if i == max_idx:
                value_str = '**' + value_str + '**'
            value_strs.append(value_str)
        print('| ' + ' | '.join([task_name] + value_strs) + ' |')


def class_entropy(classes, weights):
    counts = np.array([np.sum((np.array(classes) == class_num).astype('float32') * weights)
                       for class_num in set(classes)])
    probs = counts / np.sum(counts)
    return np.negative(np.sum(np.log(probs) * probs))


if __name__ == '__main__':
    main()
