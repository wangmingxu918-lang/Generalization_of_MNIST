
import tensorflow as tf

from experiments.common import run_multiple_seeds


def rotation_augmentation():
    return tf.keras.layers.RandomRotation(
        factor=10 / 360,
        fill_mode="constant",
        fill_value=0.0
    )


run_multiple_seeds(
    name="Rotation",
    augmentation_factory=rotation_augmentation,
    seeds=[42, 123, 2025, 7, 99]
)
