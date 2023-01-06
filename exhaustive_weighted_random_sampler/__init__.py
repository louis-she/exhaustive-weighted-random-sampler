import torch
from torch.utils.data.sampler import WeightedRandomSampler
from typing import Iterator, Sequence


class ExhaustiveWeightedRandomSampler(WeightedRandomSampler):
    """ExhaustiveWeightedRandomSampler behaves pretty much the same as WeightedRandomSampler
    except that it receives an extra parameter, exaustive_weight, which is the weight of the
    elements that should be sampled exhaustively over multiple iterations.

    This is useful when the dataset is very big and also very imbalanced, like the negative
    sample is way more than positive samples, we want to over sample positive ones, but also
    iterate over all the negative samples as much as we can.

    Args:
        weights (sequence): a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        exaustive_weight (int): which weight of samples should be sampled exhaustively,
            normally this is the one that should not been over sampled, like the lowest
            weight of samples in the dataset.
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        exaustive_weight=1,
        generator=None,
    ) -> None:
        super().__init__(weights, num_samples, True, generator)
        self.all_indices = torch.tensor(list(range(num_samples)))
        self.exaustive_weight = exaustive_weight
        self.weights_mapping = torch.tensor(weights) == self.exaustive_weight
        self.remaining_indices = torch.tensor([], dtype=torch.long)

    def get_remaining_indices(self) -> torch.Tensor:
        remaining_indices = self.weights_mapping.nonzero().squeeze()
        return remaining_indices[torch.randperm(len(remaining_indices))]

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=self.generator
        )
        exaustive_indices = rand_tensor[
            self.weights_mapping[rand_tensor].nonzero().squeeze()
        ]
        while len(exaustive_indices) > len(self.remaining_indices):
            self.remaining_indices = torch.cat(
                [self.remaining_indices, self.get_remaining_indices()]
            )
        yield_indexes, self.remaining_indices = (
            self.remaining_indices[: len(exaustive_indices)],
            self.remaining_indices[len(exaustive_indices) :],
        )
        rand_tensor[
            (rand_tensor[..., None] == exaustive_indices).any(-1).nonzero().squeeze()
        ] = yield_indexes
        yield from iter(rand_tensor.tolist())
