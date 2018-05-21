# data-balance

In this repo, I'm going to explore various kinds of dataset balancing techniques.

# Problem specification

In this problem, we get to see the entire MNIST training set (without labels), which we can use however we like. Then we are given an unbalanced subset of the test set. For example, we might be given 1000 2s and 100 3s. Our goal is to use our knowledge from the training set to re-weight the samples from the test set, such that all of the classes have a similar total weight.

# Results

The balancers don't work as well as I'd hoped, but they are almost all significantly better than random. The following table shows different tasks in each row, and gives the reweighted class entropy of the result. The class entropy measures how many bits of information it takes, on average, to encode the class of a sample from the reweighted distribution. For two classes, the maximum is ln(2) ~= 0.69.

| Task | uniform | density | voronoi | box_voronoi | smooth_voronoi | cluster | kde |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2, 3 (balanced) | **0.693** | 0.686 | 0.693 | 0.684 | 0.682 | 0.679 | 0.690 |
| 2 (x2), 3 (x1) | 0.638 | 0.665 | **0.692** | 0.678 | 0.663 | 0.655 | 0.692 |
| 5 (10%), 1 (90%) | 0.275 | 0.594 | **0.687** | 0.650 | 0.554 | 0.293 | 0.675 |
| 7 (10%), 8 (90%) | 0.361 | **0.691** | 0.485 | 0.666 | 0.686 | 0.531 | 0.376 |
| 8 (10%), 7 (90%) | 0.291 | 0.535 | 0.669 | 0.432 | 0.570 | 0.334 | **0.691** |
| 2 (10%), 3 (90%) | 0.320 | 0.612 | 0.516 | 0.596 | **0.624** | 0.357 | 0.515 |
| 3 (10%), 2 (90%) | 0.326 | 0.496 | **0.548** | 0.415 | 0.518 | 0.482 | 0.531 |
| 6 (30%), 9 (70%) | 0.613 | **0.689** | 0.609 | 0.688 | 0.685 | 0.597 | 0.563 |
| 4 (30%), 9 (70%) | 0.624 | 0.681 | 0.578 | **0.693** | 0.686 | 0.607 | 0.481 |
