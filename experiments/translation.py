
from experiments.common import run_multiple_seeds
import tensorflow as tf


def translation_augmentation():
    return tf.keras.layers.RandomTranslation(
        height_factor=0.10,
        width_factor=0.10,
        fill_mode="constant",
        fill_value=0.0
    )


run_multiple_seeds(
    name="Translation",
    augmentation_factory=translation_augmentation,
    seeds=[42, 123, 2025, 7, 99]
)
