# data-balance

In this repo, I'm going to explore various kinds of dataset balancing techniques.

# Problem specification

In this problem, we get to see the entire MNIST training set (without labels), which we can use however we like. Then we are given an unbalanced subset of the test set. For example, we might be given 1000 2s and 100 3s. Our goal is to use our knowledge from the training set to re-weight the samples from the test set, such that all of the classes have a similar total weight.

# Results

The balancers don't work as well as I'd hoped, but they are almost all significantly better than random. The following table shows different tasks in each row, and gives the reweighted class distribution of the result.

| Task | uniform | density | voronoi | box_voronoi | smooth_voronoi | cluster | kde |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2<br>3 (balanced) | 2: 0.497<br>3: 0.503 | 2: 0.561<br>3: 0.439 | 2: 0.478<br>3: 0.522 | 2: 0.559<br>3: 0.441 | 2: 0.557<br>3: 0.443 | 2: 0.493<br>3: 0.507 | 2: 0.461<br>3: 0.539 |
| 2 (x2)<br>3 (x1) | 2: 0.664<br>3: 0.336 | 2: 0.619<br>3: 0.381 | 2: 0.502<br>3: 0.498 | 2: 0.551<br>3: 0.449 | 2: 0.615<br>3: 0.385 | 2: 0.642<br>3: 0.358 | 2: 0.526<br>3: 0.474 |
| 5 (10%)<br>1 (90%) | 1: 0.922<br>5: 0.078 | 1: 0.714<br>5: 0.286 | 1: 0.605<br>5: 0.395 | 1: 0.721<br>5: 0.279 | 1: 0.765<br>5: 0.235 | 1: 0.840<br>5: 0.160 | 1: 0.482<br>5: 0.518 |
| 2 (10%)<br>3 (90%) | 2: 0.098<br>3: 0.902 | 2: 0.299<br>3: 0.701 | 2: 0.244<br>3: 0.756 | 2: 0.248<br>3: 0.752 | 2: 0.378<br>3: 0.622 | 2: 0.147<br>3: 0.853 | 2: 0.216<br>3: 0.784 |
| 3 (10%)<br>2 (90%) | 2: 0.900<br>3: 0.100 | 2: 0.777<br>3: 0.223 | 2: 0.740<br>3: 0.260 | 2: 0.840<br>3: 0.160 | 2: 0.727<br>3: 0.273 | 2: 0.831<br>3: 0.169 | 2: 0.734<br>3: 0.266 |
