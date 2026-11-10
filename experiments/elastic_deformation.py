
from experiments.common import run_multiple_seeds
from augmentations.elastic import ElasticDeformation


# Use the same five seeds for every experiment.
seeds = [42, 123, 2025, 7, 99]


def weak_augmentation():
    return ElasticDeformation(
        max_displacement=1.5,
        grid_size=4
    )


def moderate_augmentation():
    return ElasticDeformation(
        max_displacement=3.0,
        grid_size=4
    )


# Experiment 1: Weak elastic deformation
run_multiple_seeds(
    name="Elastic_Weak",
    augmentation_factory=weak_augmentation,
    seeds=seeds
)


# Experiment 2: Moderate elastic deformation
run_multiple_seeds(
    name="Elastic_Moderate",
    augmentation_factory=moderate_augmentation,
    seeds=seeds
)
