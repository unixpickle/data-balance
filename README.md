# data-balance

In this repo, I'm going to explore various kinds of dataset balancing techniques.

# Problem specification

In this problem, we get to see the entire MNIST training set (without labels), which we can use however we like. Then we are given an unbalanced subset of the test set. For example, we might be given 1000 2s and 100 3s. Our goal is to use our knowledge from the training set to re-weight the samples from the test set, such that all of the classes have a similar total weight.

# Results

The balancers don't work as well as I'd hoped, but they are almost all significantly better than random. The following table shows different tasks in each row, and gives the reweighted class entropy of the result. The class entropy measures how many bits of information it takes, on average, to encode the class of a sample from the reweighted distribution. For two classes, the maximum is ln(2) ~= 0.69.

| Task | uniform | density | voronoi | box_voronoi | smooth_voronoi | cluster | kde |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2, 3 (balanced) | **0.693** | 0.686 | 0.693 | 0.685 | 0.689 | 0.692 | 0.690 |
| 2 (x2), 3 (x1) | 0.638 | 0.665 | **0.692** | 0.684 | 0.673 | 0.599 | 0.692 |
| 5 (10%), 1 (90%) | 0.275 | 0.555 | **0.687** | 0.641 | 0.496 | 0.471 | 0.675 |
| 7 (10%), 8 (90%) | 0.361 | **0.687** | 0.500 | 0.664 | 0.686 | 0.520 | 0.413 |
| 8 (10%), 7 (90%) | 0.291 | 0.513 | 0.621 | 0.527 | 0.577 | 0.329 | **0.681** |
| 2 (10%), 3 (90%) | 0.320 | 0.603 | 0.485 | 0.491 | **0.668** | 0.404 | 0.492 |
| 3 (10%), 2 (90%) | 0.326 | 0.518 | **0.605** | 0.449 | 0.564 | 0.471 | 0.602 |
| 6 (30%), 9 (70%) | 0.613 | 0.689 | 0.609 | 0.686 | **0.689** | 0.609 | 0.569 |
| 4 (30%), 9 (70%) | 0.624 | 0.689 | 0.561 | **0.692** | 0.691 | 0.680 | 0.499 |
| mean improvement | 0.000 | 0.163 | 0.146 | 0.153 | **0.177** | 0.071 | 0.130 |
