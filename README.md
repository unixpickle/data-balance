# data-balance

In this repo, I'm going to explore various kinds of dataset balancing techniques.

# Problem specification

In this problem, we get to see the entire MNIST training set (without labels), which we can use however we like. Then we are given an unbalanced subset of the test set. For example, we might be given 1000 2s and 100 3s. Our goal is to use our knowledge from the training set to re-weight the samples from the test set, such that all of the classes have a similar total weight.

# Results

The balancers don't work as well as I'd hoped, but they are almost all significantly better than random. The following table shows different tasks in each row, and gives the reweighted class distribution of the result.

| Task | uniform | density | voronoi | box_voronoi | smooth_voronoi | cluster | kde |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2, 3 (balanced) | 2: 0.497<br>3: 0.503 | 2: 0.561<br>3: 0.439 | 2: 0.489<br>3: 0.511 | 2: 0.540<br>3: 0.460 | 2: 0.555<br>3: 0.445 | 2: 0.507<br>3: 0.493 | 2: 0.461<br>3: 0.539 |
| 2 (x2), 3 (x1) | 2: 0.664<br>3: 0.336 | 2: 0.619<br>3: 0.381 | 2: 0.522<br>3: 0.478 | 2: 0.575<br>3: 0.425 | 2: 0.632<br>3: 0.368 | 2: 0.692<br>3: 0.308 | 2: 0.526<br>3: 0.474 |
| 5 (10%), 1 (90%) | 1: 0.922<br>5: 0.078 | 1: 0.765<br>5: 0.235 | 1: 0.532<br>5: 0.468 | 1: 0.625<br>5: 0.375 | 1: 0.819<br>5: 0.181 | 1: 0.926<br>5: 0.074 | 1: 0.409<br>5: 0.591 |
| 7 (10%), 8 (90%) | 7: 0.117<br>8: 0.883 | 7: 0.467<br>8: 0.533 | 7: 0.240<br>8: 0.760 | 7: 0.323<br>8: 0.677 | 7: 0.497<br>8: 0.503 | 7: 0.207<br>8: 0.793 | 7: 0.155<br>8: 0.845 |
| 8 (10%), 7 (90%) | 7: 0.915<br>8: 0.085 | 7: 0.809<br>8: 0.191 | 7: 0.525<br>8: 0.475 | 7: 0.793<br>8: 0.207 | 7: 0.749<br>8: 0.251 | 7: 0.897<br>8: 0.103 | 7: 0.456<br>8: 0.544 |
| 2 (10%), 3 (90%) | 2: 0.098<br>3: 0.902 | 2: 0.321<br>3: 0.679 | 2: 0.226<br>3: 0.774 | 2: 0.265<br>3: 0.735 | 2: 0.388<br>3: 0.612 | 2: 0.143<br>3: 0.857 | 2: 0.193<br>3: 0.807 |
| 3 (10%), 2 (90%) | 2: 0.900<br>3: 0.100 | 2: 0.806<br>3: 0.194 | 2: 0.746<br>3: 0.254 | 2: 0.848<br>3: 0.152 | 2: 0.743<br>3: 0.257 | 2: 0.902<br>3: 0.098 | 2: 0.708<br>3: 0.292 |
| 6 (30%), 9 (70%) | 6: 0.302<br>9: 0.698 | 6: 0.434<br>9: 0.566 | 6: 0.274<br>9: 0.726 | 6: 0.435<br>9: 0.565 | 6: 0.441<br>9: 0.559 | 6: 0.398<br>9: 0.602 | 6: 0.228<br>9: 0.772 |
| 4 (30%), 9 (70%) | 4: 0.316<br>9: 0.684 | 4: 0.425<br>9: 0.575 | 4: 0.251<br>9: 0.749 | 4: 0.526<br>9: 0.474 | 4: 0.413<br>9: 0.587 | 4: 0.382<br>9: 0.618 | 4: 0.199<br>9: 0.801 |
