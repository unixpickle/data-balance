# data-balance

In this repo, I'm going to explore various kinds of dataset balancing techniques.

# Problem specification

In this problem, we get to see the entire MNIST training set (without labels), which we can use however we like. Then we are given an unbalanced subset of the test set. For example, we might be given 1000 2s and 100 3s. Our goal is to use our knowledge from the training set to re-weight the samples from the test set, such that all of the classes have a similar total weight.

# Results

The balancers don't work as well as I'd hoped, but they are almost all significantly better than random. The following table shows different tasks in each row, and gives the reweighted class distribution of the result.

| Task | uniform | density | voronoi | box_voronoi | smooth_voronoi | cluster | kde |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2, 3 (balanced) | 2: 0.497<br>3: 0.503 | 2: 0.561<br>3: 0.439 | 2: 0.523<br>3: 0.477 | 2: 0.558<br>3: 0.442 | 2: 0.565<br>3: 0.435 | 2: 0.500<br>3: 0.500 | 2: 0.461<br>3: 0.539 |
| 2 (x2), 3 (x1) | 2: 0.664<br>3: 0.336 | 2: 0.619<br>3: 0.381 | 2: 0.500<br>3: 0.500 | 2: 0.562<br>3: 0.438 | 2: 0.635<br>3: 0.365 | 2: 0.690<br>3: 0.310 | 2: 0.526<br>3: 0.474 |
| 5 (10%), 1 (90%) | 1: 0.922<br>5: 0.078 | 1: 0.766<br>5: 0.234 | 1: 0.559<br>5: 0.441 | 1: 0.650<br>5: 0.350 | 1: 0.740<br>5: 0.260 | 1: 0.928<br>5: 0.072 | 1: 0.474<br>5: 0.526 |
| 2 (10%), 3 (90%) | 2: 0.098<br>3: 0.902 | 2: 0.341<br>3: 0.659 | 2: 0.226<br>3: 0.774 | 2: 0.297<br>3: 0.703 | 2: 0.389<br>3: 0.611 | 2: 0.189<br>3: 0.811 | 2: 0.254<br>3: 0.746 |
| 3 (10%), 2 (90%) | 2: 0.900<br>3: 0.100 | 2: 0.797<br>3: 0.203 | 2: 0.738<br>3: 0.262 | 2: 0.861<br>3: 0.139 | 2: 0.744<br>3: 0.256 | 2: 0.803<br>3: 0.197 | 2: 0.724<br>3: 0.276 |
