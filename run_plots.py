"""
Generate a plot of VAE features on an imbalanced dataset.
"""

from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA

from data_balance.data import balancing_task
from data_balance.vae import vae_features

VAE_CHECKPOINT = 'vae_checkpoint'


def main():
    images, classes = balancing_task([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1.0] * 10)

    image_pca = run_pca(images)
    features, _ = vae_features(images)
    feature_pca = run_pca(features)

    pyplot.figure(1)
    pyplot.scatter(feature_pca[:, 0], feature_pca[:, 1], c=list(classes))
    pyplot.show()

    pyplot.figure(2)
    pyplot.scatter(image_pca[:, 0], image_pca[:, 1], c=list(classes))
    pyplot.show()


def run_pca(data):
    data = np.reshape(data, [data.shape[0], -1])
    pca = PCA(n_components=2)
    pca.fit(data)
    return pca.transform(data)


if __name__ == '__main__':
    main()
