
from experiments.common import run_multiple_seeds
import tensorflow as tf


def flipping_augmentation():
    return tf.keras.layers.RandomFlip(
        mode="horizontal"
    )


run_multiple_seeds(
    name="Flipping",
    augmentation_factory=flipping_augmentation,
    seeds=[42, 123, 2025, 7, 99]
)
