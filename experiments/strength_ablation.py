
import tensorflow as tf

from experiments.common import run_multiple_seeds


seeds = [42, 123, 2025, 7, 99]


# Rotation: compare three magnitudes
for degrees in [5, 10, 15]:

    def augmentation_factory(degrees=degrees):
        return tf.keras.layers.RandomRotation(
            factor=degrees / 360,
            fill_mode="constant",
            fill_value=0.0
        )

    run_multiple_seeds(
        name=f"Rotation_{degrees}_degrees",
        augmentation_factory=augmentation_factory,
        seeds=seeds
    )


# Translation: compare three magnitudes
for amount in [0.05, 0.10, 0.15]:

    def augmentation_factory(amount=amount):
        return tf.keras.layers.RandomTranslation(
            height_factor=amount,
            width_factor=amount,
            fill_mode="constant",
            fill_value=0.0
        )

    run_multiple_seeds(
        name=f"Translation_{int(amount * 100)}_percent",
        augmentation_factory=augmentation_factory,
        seeds=seeds
    )


# Cropping: compare three crop sizes
for crop_size in [26, 24, 22]:

    def augmentation_factory(crop_size=crop_size):
        return tf.keras.Sequential([
            tf.keras.layers.RandomCrop(
                height=crop_size,
                width=crop_size
            ),
            tf.keras.layers.Resizing(
                height=28,
                width=28
            )
        ])

    run_multiple_seeds(
        name=f"Cropping_{crop_size}x{crop_size}",
        augmentation_factory=augmentation_factory,
        seeds=seeds
    )
