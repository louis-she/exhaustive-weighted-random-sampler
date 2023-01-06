# ExhaustiveWeightedRandomSampler
[![run test](https://github.com/louis-she/exhaustive-weighted-random-sampler/actions/workflows/test.yaml/badge.svg)](https://github.com/louis-she/exhaustive-weighted-random-sampler/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/louis-she/exhaustive-weighted-random-sampler/branch/main/graph/badge.svg?token=MMZ4PEB1Y7)](https://codecov.io/gh/louis-she/exhaustive-weighted-random-sampler)

ExhaustiveWeightedRandomSampler can exhaustively sample the indices with a specific weight over epochs.

## Installation

```
pip install git+https://github.com/louis-she/exhaustive-weighted-random-sampler.git
```

## Comparing the differences

```python
import torch
from torch.utils.data import WeightedRandomSampler
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler

sampler = WeightedRandomSampler([1, 1, 1, 1, 1, 1, 1, 1, 1, 10], num_samples=5)
for i in range(5):
    print(list(sampler))

"""
output:
[4, 3, 9, 3, 4]
[0, 5, 0, 9, 8]
[9, 9, 0, 9, 2]
[9, 9, 7, 9, 9]
[9, 9, 9, 9, 9]

explain: there are no 1 and 6, but 0 appears three times
"""

sampler = ExhaustiveWeightedRandomSampler([1, 1, 1, 1, 1, 1, 1, 1, 1, 10], num_samples=5)
for i in range(5):
    print(list(sampler))

"""
output:
[4, 6, 9, 9, 9]
[1, 0, 9, 9, 5]
[9, 7, 3, 8, 9]
[9, 2, 1, 9, 9]
[8, 9, 7, 3, 2]

explain: all the 0 to 8 appears in the yield results.
"""
```
