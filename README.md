# data-balance

In this repo, I'm going to explore various kinds of dataset balancing techniques.

# Problem specification

In this problem, we get to see the entire MNIST training set (without labels), which we can use however we like. Then we are given an unbalanced subset of the test set. For example, we might be given 1000 2s and 100 3s. Our goal is to use our knowledge from the training set to re-weight the samples from the test set, such that all of the classes have a similar total weight.

# Results

The balancers don't work as well as I'd hoped, but they are almost all significantly better than random. The following table shows different tasks in each row, and gives the reweighted class distribution of the result.

| Task | uniform | density | voronoi | box_voronoi | smooth_voronoi | cluster | kde |
| 2, 3 (balanced) | {2: 0.49745157, 3: 0.50254834} | {2: 0.56067955, 3: 0.43932047} | {2: 0.51580024, 3: 0.48419976} | {2: 0.56371045, 3: 0.4362895} | {2: 0.55657494, 3: 0.4434251} | {2: 0.50807888265627, 3: 0.49192111734373023} | {2: 0.46079313932953353, 3: 0.5392068606704665} |
| 2 (x2), 3 (x1) | {2: 0.6643976, 3: 0.33560246} | {2: 0.61866564, 3: 0.3813344} | {2: 0.496256, 3: 0.50374407} | {2: 0.5643295, 3: 0.43567055} | {2: 0.6272863, 3: 0.37271369} | {2: 0.6929830969034367, 3: 0.30701690309656315} | {2: 0.526457271198997, 3: 0.4735427288010029} |
| 5 (10%), 1 (90%) | {1: 0.9216758, 5: 0.07832423} | {1: 0.71073204, 5: 0.28926802} | {1: 0.51912564, 5: 0.48087427} | {1: 0.62841535, 5: 0.3715847} | {1: 0.7319368, 5: 0.26806313} | {1: 0.9173076923076924, 5: 0.08269230769230768} | {1: 0.4117148166296073, 5: 0.5882851833703927} |
| 2 (10%), 3 (90%) | {2: 0.09775968, 3: 0.9022405} | {2: 0.29095468, 3: 0.7090454} | {2: 0.19959268, 3: 0.80040735} | {2: 0.2382892, 3: 0.76171076} | {2: 0.36863542, 3: 0.6313646} | {2: 0.14666666666666667, 3: 0.8533333333333335} | {2: 0.18532763861170054, 3: 0.8146723613882996} |
| 3 (10%), 2 (90%) | {2: 0.89958996, 3: 0.10040982} | {2: 0.77541983, 3: 0.22458015} | {2: 0.7008197, 3: 0.29918033} | {2: 0.8114754, 3: 0.18852459} | {2: 0.71967214, 3: 0.28032786} | {2: 0.8910539215686275, 3: 0.10894607843137254} | {2: 0.6782522797769561, 3: 0.32174772022304376} |
