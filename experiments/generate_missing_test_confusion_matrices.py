
import os
import glob

import numpy as np
import tensorflow as tf

from data_preprocessing import (
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test
)

from model import build_model
from augmentations.elastic import ElasticDeformation
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. Combine training and validation datasets
# --------------------------------------------------

x_full_train = np.concatenate([x_train, x_val], axis=0)
y_full_train = np.concatenate([y_train, y_val], axis=0)

print("Full training dataset:", x_full_train.shape)
print("Test dataset:", x_test.shape)


# --------------------------------------------------
# 2. Define augmentation factories
# --------------------------------------------------

def rotation_augmentation(degrees):
    return tf.keras.layers.RandomRotation(
        factor=degrees / 360,
        fill_mode="constant",
        fill_value=0.0
    )


def translation_augmentation(amount):
    return tf.keras.layers.RandomTranslation(
        height_factor=amount,
        width_factor=amount,
        fill_mode="constant",
        fill_value=0.0
    )


def cropping_augmentation(crop_size):
    return tf.keras.Sequential([
        tf.keras.layers.RandomCrop(
            height=crop_size,
            width=crop_size
        ),
        tf.keras.layers.Resizing(28, 28)
    ])


def flipping_augmentation():
    return tf.keras.layers.RandomFlip(
        mode="horizontal"
    )


def elastic_augmentation(displacement):
    return ElasticDeformation(
        max_displacement=displacement,
        grid_size=4
    )


def gaussian_noise_augmentation(sigma):
    return tf.keras.Sequential([
        tf.keras.layers.GaussianNoise(stddev=sigma),
        tf.keras.layers.Lambda(
            lambda images: tf.clip_by_value(
                images, 0.0, 1.0
            )
        )
    ])


# --------------------------------------------------
# 3. Define all configurations
# --------------------------------------------------

experiments = [
    ("Baseline", None),

    ("Rotation_10_degrees",
     lambda: rotation_augmentation(10)),

    ("Rotation_5_degrees",
     lambda: rotation_augmentation(5)),

    ("Rotation_15_degrees",
     lambda: rotation_augmentation(15)),

    ("Translation_10_percent",
     lambda: translation_augmentation(0.10)),

    ("Translation_5_percent",
     lambda: translation_augmentation(0.05)),

    ("Translation_15_percent",
     lambda: translation_augmentation(0.15)),

    ("Cropping_24x24",
     lambda: cropping_augmentation(24)),

    ("Cropping_26x26",
     lambda: cropping_augmentation(26)),

    ("Cropping_22x22",
     lambda: cropping_augmentation(22)),

    ("Flipping",
     flipping_augmentation),

    ("Elastic_Weak",
     lambda: elastic_augmentation(1.5)),

    ("Elastic_Moderate",
     lambda: elastic_augmentation(3.0)),

    ("Gaussian_Noise_0.05",
     lambda: gaussian_noise_augmentation(0.05)),

    ("Gaussian_Noise_0.1",
     lambda: gaussian_noise_augmentation(0.10)),

    ("Gaussian_Noise_0.15",
     lambda: gaussian_noise_augmentation(0.15))
]


# --------------------------------------------------
# 4. Find an existing test confusion matrix
# --------------------------------------------------

def existing_test_matrix(name, seed):

    files = glob.glob(
        "results/**/*.csv",
        recursive=True
    )

    for path in files:

        filename = os.path.basename(path).lower()

        if "confusion" not in filename:
            continue

        if "matrix" not in filename:
            continue

        if f"seed_{seed}" not in filename:
            continue

        if name.lower() not in filename:
            continue

        try:
            matrix = np.loadtxt(
                path,
                delimiter=",",
                dtype=int
            )

            if matrix.shape == (10, 10) and matrix.sum() == len(y_test):
                return path

        except (ValueError, OSError):
            continue

    return None


# --------------------------------------------------
# 5. Generate only missing test matrices
# --------------------------------------------------

seeds = [42, 123, 2025, 7, 99]

output_dir = "results/test_confusion_matrices"
os.makedirs(output_dir, exist_ok=True)

generated = 0
skipped = 0

for name, augmentation_factory in experiments:

    for seed in seeds:

        print("\n" + "=" * 60)
        print(f"Experiment: {name}")
        print(f"Seed: {seed}")
        print("=" * 60)

        existing_path = existing_test_matrix(name, seed)

        if existing_path is not None:

            print("Existing test matrix found:")
            print(existing_path)
            print("Skipping this experiment.")

            skipped += 1
            continue

        print("No valid test matrix found. Training model...")

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        augmentation = (
            augmentation_factory()
            if augmentation_factory is not None
            else None
        )

        model = build_model(
            augmentation=augmentation
        )

        model.fit(
            x_full_train,
            y_full_train,
            epochs=10,
            batch_size=128,
            verbose=1
        )

        predictions = model.predict(
            x_test,
            batch_size=128,
            verbose=0
        )

        predicted_labels = np.argmax(
            predictions,
            axis=1
        )

        cm = confusion_matrix(
            y_test,
            predicted_labels,
            labels=np.arange(10)
        )

        filename = (
            f"{name}_seed_{seed}_test_confusion_matrix.csv"
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        np.savetxt(
            output_path,
            cm,
            delimiter=",",
            fmt="%d"
        )

        print("Saved:", output_path)
        generated += 1


# --------------------------------------------------
# 6. Final report
# --------------------------------------------------

print("\n" + "=" * 60)
print("CONFUSION MATRIX GENERATION COMPLETED")
print("=" * 60)

print("Existing matrices skipped:", skipped)
print("New matrices generated:", generated)
print("Total matrices:", skipped + generated)
