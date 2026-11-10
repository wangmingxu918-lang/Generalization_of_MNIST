
from experiments.common import run_multiple_seeds
import tensorflow as tf


def cropping_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomCrop(
            height=24,
            width=24
        ),
        tf.keras.layers.Resizing(
            height=28,
            width=28
        )
    ])


run_multiple_seeds(
    name="Cropping",
    augmentation_factory=cropping_augmentation,
    seeds=[42, 123, 2025, 7, 99]
)
