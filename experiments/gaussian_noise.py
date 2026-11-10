
import tensorflow as tf

from experiments.common import run_multiple_seeds


seeds = [42, 123, 2025, 7, 99]


for sigma in [0.05, 0.10, 0.15]:

    def augmentation_factory(sigma=sigma):
        return tf.keras.Sequential([
            tf.keras.layers.GaussianNoise(
                stddev=sigma
            ),
            tf.keras.layers.Lambda(
                lambda images: tf.clip_by_value(
                    images, 0.0, 1.0
                )
            )
        ])

    run_multiple_seeds(
        name=f"Gaussian_Noise_{sigma}",
        augmentation_factory=augmentation_factory,
        seeds=seeds
    )
