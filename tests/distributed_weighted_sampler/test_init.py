import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from ignite.distributed import DistributedProxySampler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler


@pytest.fixture
def dataset01():
    class _Dataset(Dataset):
        def __init__(self):
            self.data = [1] * 10000 + [0] * 90000

        def __getitem__(self, index):
            return torch.tensor(self.data[index])

        def __len__(self):
            return len(self.data)

    return _Dataset()


@pytest.fixture
def dataset_sequence():
    class _Dataset(Dataset):
        def __init__(self):
            self.data = list(range(100000))

        def __getitem__(self, index):
            return torch.tensor(self.data[index])

        def __len__(self):
            return len(self.data)

    return _Dataset()


@pytest.fixture
def weights():
    return [9] * 10000 + [1] * 90000


def test_deterministic(weights):
    torch.manual_seed(777)
    indices_1 = list(ExhaustiveWeightedRandomSampler(weights, num_samples=100))
    indices_2 = list(ExhaustiveWeightedRandomSampler(weights, num_samples=100))

    torch.manual_seed(777)
    indices_3 = list(ExhaustiveWeightedRandomSampler(weights, num_samples=100))

    assert all([i != j for i, j in zip(indices_1, indices_2)])
    assert all([i != j for i, j in zip(indices_2, indices_3)])
    assert all([i == j for i, j in zip(indices_1, indices_3)])


@pytest.mark.parametrize(
    "klass", [WeightedRandomSampler, ExhaustiveWeightedRandomSampler]
)
def test_weights_should_work_as_weighted_random_sampler(klass, dataset01, weights):
    sampler = klass(weights, num_samples=10000)
    loader = DataLoader(dataset01, sampler=sampler, num_workers=2, batch_size=1000)
    results = []
    for _ in range(10):
        for batch in loader:
            results.append(batch)
    results = torch.cat(results)
    assert results.shape[0] == 100000
    assert (results == 1).sum().item() == pytest.approx(50000, abs=500)
    assert (results == 0).sum().item() == pytest.approx(50000, abs=500)


@pytest.mark.parametrize(
    "klass", [WeightedRandomSampler, ExhaustiveWeightedRandomSampler]
)
def test_no_duplicates_in_less_than_one_around(klass, dataset_sequence, weights):
    sampler = klass(weights, num_samples=10000)
    loader = DataLoader(dataset_sequence, sampler=sampler, num_workers=2, batch_size=32)
    results = []
    for _ in range(10):
        for batch in loader:
            results.append(batch)
    results = torch.cat(results)
    weight1_indices = results[results >= 10000]
    if klass == WeightedRandomSampler:
        # since there should be some duplicates
        assert len(weight1_indices) > len(weight1_indices.unique())
    elif klass == ExhaustiveWeightedRandomSampler:
        # since there should be no duplicates
        assert len(weight1_indices) == len(weight1_indices.unique())


@pytest.mark.parametrize(
    "klass", [WeightedRandomSampler, ExhaustiveWeightedRandomSampler]
)
@pytest.mark.parametrize("round", [2, 3, 4])
def test_one_duplicates_at_most_than_one_around(
    klass, round, dataset_sequence, weights
):
    """for every 10000 samples, there should be approximately 5000 weight_1 and 5000
    weight_9 samples, so if we want to go over all the 90000 weight_1 samples, we should
    go like 90000 / 5000 = 18 times over the 10000 samples, so a round should be 18 epochs.
    But since there are some randomness, we will set 17 epochs as one round.
    """
    sampler = klass(weights, num_samples=10000)
    loader = DataLoader(
        dataset_sequence, sampler=sampler, num_workers=2, batch_size=1000
    )
    results = []
    round_epochs = 17
    for _ in range(round * round_epochs):
        for batch in loader:
            results.append(batch)
    results = torch.cat(results)
    weight1_indices = results[results >= 10000]
    if klass == WeightedRandomSampler:
        # there should be duplicates more than 2 times
        assert weight1_indices.unique(return_counts=True)[1].max().item() > round
    elif klass == ExhaustiveWeightedRandomSampler:
        # since there should duplicates at least 2 times
        assert weight1_indices.unique(return_counts=True)[1].max().item() == round


@pytest.mark.distributed
def test_distributed_sampler(weights, worker_id):
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://localhost:32123",
        world_size=2,
        rank=int(worker_id[-1]),
        timeout=timedelta(seconds=10),
    )
    dist.barrier()
    sampler = DistributedProxySampler(
        ExhaustiveWeightedRandomSampler(weights, num_samples=10000)
    )
    sampled_indices = torch.tensor(list(sampler))

    assert len(sampled_indices) == 5000
    assert (
        sampled_indices[sampled_indices >= 10000]
        .unique(return_counts=True)[1]
        .max()
        .item()
        == 1
    )
    tensor_list = [torch.zeros(5000, dtype=torch.long) for _ in range(2)]
    dist.all_gather(tensor_list, sampled_indices)

    all_indices = torch.cat(tensor_list, dim=0)
    assert len(all_indices) == 10000
    assert (
        all_indices[all_indices >= 10000].unique(return_counts=True)[1].max().item()
        == 1
    )
    dist.destroy_process_group()
    time.sleep(1)


@pytest.mark.distributed
def test_distributed_sampler_and_local_sampler_should_match(weights, worker_id):
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://localhost:32123",
        world_size=2,
        rank=int(worker_id[-1]),
        timeout=timedelta(seconds=10),
    )
    dist.barrier()

    sampler = DistributedProxySampler(
        ExhaustiveWeightedRandomSampler(weights, num_samples=10000)
    )
    sampled_indices = torch.tensor(list(sampler))
    tensor_list = [torch.zeros(5000, dtype=torch.long) for _ in range(2)]
    dist.all_gather(tensor_list, sampled_indices)
    all_indices_distributed = torch.cat(tensor_list, dim=0).tolist()

    # distributed sampler will always set the seed to 0
    torch.manual_seed(0)
    all_indices_local = list(
        ExhaustiveWeightedRandomSampler(weights, num_samples=10000)
    )
    assert all(
        [
            x == y
            for x, y in zip(sorted(all_indices_distributed), sorted(all_indices_local))
        ]
    )
    dist.destroy_process_group()
    time.sleep(1)


@pytest.mark.distributed
@pytest.mark.parametrize("round", [2, 3, 4])
def test_distributed_sampler_in_multiple_round(
    round, weights, dataset_sequence, worker_id
):
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://localhost:32123",
        world_size=2,
        rank=int(worker_id[-1]),
        timeout=timedelta(seconds=10),
    )
    dist.barrier()

    sampler = DistributedProxySampler(
        ExhaustiveWeightedRandomSampler(weights, num_samples=10000)
    )
    loader = DataLoader(
        dataset_sequence, sampler=sampler, num_workers=2, batch_size=1000
    )
    results = []
    round_epochs = 17
    for _ in range(round * round_epochs):
        for batch in loader:
            results.append(batch)
    results = torch.cat(results)
    weight1_indices = results[results >= 10000]
    assert weight1_indices.unique(return_counts=True)[1].max().item() == round
    dist.destroy_process_group()
    time.sleep(1)


def test_side_cases_1():
    indices = list(ExhaustiveWeightedRandomSampler([1, 1, 1, 1, 1], num_samples=10))
    assert len(indices) == 10
    assert torch.tensor(indices).unique(return_counts=True)[1].max().item() == 2
