# data-balance

In this repo, I'm going to explore various kinds of dataset balancing techniques.

# Problem specification

In this problem, we get to see the entire MNIST training set (without labels), which we can use however we like. Then we are given an unbalanced subset of the test set. For example, we might be given 1000 2s and 100 3s. Our goal is to use our knowledge from the training set to re-weight the samples from the test set, such that all of the classes have a similar total weight.

# Results

The balancers don't work as well as I'd hoped, but they are almost all significantly better than random. The following table shows different tasks in each row, and gives the reweighted class entropy of the result. The class entropy measures how many bits of information it takes, on average, to encode the class of a sample from the reweighted distribution. For two classes, the maximum is ln(2) ~= 0.69.

| Task | uniform | density | voronoi | box_voronoi | smooth_voronoi | cluster | kde |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2, 3 (balanced) | **0.693** | 0.686 | 0.693 | 0.683 | 0.686 | 0.678 | 0.690 |
| 2 (x2), 3 (x1) | 0.638 | 0.665 | **0.693** | 0.688 | 0.660 | 0.633 | 0.692 |
| 5 (10%), 1 (90%) | 0.275 | 0.557 | **0.693** | 0.629 | 0.590 | 0.289 | 0.678 |
| 7 (10%), 8 (90%) | 0.361 | 0.693 | 0.559 | 0.652 | **0.693** | 0.675 | 0.499 |
| 8 (10%), 7 (90%) | 0.291 | 0.519 | 0.668 | 0.456 | 0.564 | 0.324 | **0.693** |
| 2 (10%), 3 (90%) | 0.320 | 0.631 | 0.508 | 0.539 | **0.693** | 0.391 | 0.421 |
| 3 (10%), 2 (90%) | 0.326 | 0.523 | 0.588 | 0.475 | 0.581 | 0.331 | **0.622** |
| 6 (30%), 9 (70%) | 0.613 | 0.687 | 0.604 | 0.684 | 0.691 | **0.692** | 0.552 |
| 4 (30%), 9 (70%) | 0.624 | 0.676 | 0.538 | **0.692** | 0.679 | 0.687 | 0.526 |
| mean improvement | 0.000 | 0.166 | 0.156 | 0.151 | 0.188 | 0.062 | 0.137 |
