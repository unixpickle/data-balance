# data-balance

In this repo<br>I'm going to explore various kinds of dataset balancing techniques.

# Problem specification

In this problem<br>we get to see the entire MNIST training set (without labels)<br>which we can use however we like. Then we are given an unbalanced subset of the test set. For example<br>we might be given 1000 2s and 100 3s. Our goal is to use our knowledge from the training set to re-weight the samples from the test set<br>such that all of the classes have a similar total weight.

# Results

The balancers don't work as well as I'd hoped<br>but they are almost all significantly better than random. The following table shows different tasks in each row<br>and gives the reweighted class distribution of the result.

| Task | uniform | density | voronoi | box_voronoi | smooth_voronoi | cluster | kde |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2, 3 (balanced) | 2: 0.497, 3: 0.503 | 2: 0.561, 3: 0.439 | 2: 0.478, 3: 0.522 | 2: 0.559, 3: 0.441 | 2: 0.557, 3: 0.443 | 2: 0.493, 3: 0.507 | 2: 0.461, 3: 0.539 |
| 2 (x2), 3 (x1) | 2: 0.664, 3: 0.336 | 2: 0.619, 3: 0.381 | 2: 0.502, 3: 0.498 | 2: 0.551, 3: 0.449 | 2: 0.615, 3: 0.385 | 2: 0.642, 3: 0.358 | 2: 0.526, 3: 0.474 |
| 5 (10%), 1 (90%) | 1: 0.922, 5: 0.078 | 1: 0.714, 5: 0.286 | 1: 0.605, 5: 0.395 | 1: 0.721, 5: 0.279 | 1: 0.765, 5: 0.235 | 1: 0.840, 5: 0.160 | 1: 0.482, 5: 0.518 |
| 2 (10%), 3 (90%) | 2: 0.098, 3: 0.902 | 2: 0.299, 3: 0.701 | 2: 0.244, 3: 0.756 | 2: 0.248, 3: 0.752 | 2: 0.378, 3: 0.622 | 2: 0.147, 3: 0.853 | 2: 0.216, 3: 0.784 |
| 3 (10%), 2 (90%) | 2: 0.900, 3: 0.100 | 2: 0.777, 3: 0.223 | 2: 0.740, 3: 0.260 | 2: 0.840, 3: 0.160 | 2: 0.727, 3: 0.273 | 2: 0.831, 3: 0.169 | 2: 0.734, 3: 0.266 |
