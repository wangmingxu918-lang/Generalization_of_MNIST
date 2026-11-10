
from experiments.common import run_multiple_seeds


seeds = [42, 123, 2025, 7, 99]


run_multiple_seeds(
    name="Baseline",
    augmentation_factory=None,
    seeds=seeds
)
