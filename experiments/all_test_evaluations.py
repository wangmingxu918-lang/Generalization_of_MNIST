
import csv
import os
import statistics

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

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
        tf.keras.layers.Resizing(
            height=28,
            width=28
        )
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
        tf.keras.layers.GaussianNoise(
            stddev=sigma
        ),
        tf.keras.layers.Lambda(
            lambda images: tf.clip_by_value(
                images, 0.0, 1.0
            )
        )
    ])


# --------------------------------------------------
# 3. Define all 16 experimental configurations
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
# 4. Training and test evaluation
# --------------------------------------------------

seeds = [42, 123, 2025, 7, 99]

os.makedirs("results", exist_ok=True)
confusion_matrix_dir = "results/test_confusion_matrices"
os.makedirs(confusion_matrix_dir, exist_ok=True)
all_results = []

for name, augmentation_factory in experiments:

    for seed in seeds:

        print("\n" + "=" * 60)
        print(f"Experiment: {name}")
        print(f"Seed: {seed}")
        print("=" * 60)

        # Reset TensorFlow state.
        tf.keras.backend.clear_session()

        # Set random seed.
        tf.keras.utils.set_random_seed(seed)

        # Create augmentation layer.
        augmentation = (
            augmentation_factory()
            if augmentation_factory is not None
            else None
        )

        # Build the existing CNN.
        model = build_model(
            augmentation=augmentation
        )

        # Train on all 60,000 training images.
        model.fit(
            x_full_train,
            y_full_train,
            epochs=10,
            batch_size=128,
            verbose=1
        )

        # Evaluate test loss.
        test_loss, _ = model.evaluate(
            x_test,
            y_test,
            batch_size=128,
            verbose=0
        )

        # Predict test labels.
        predictions = model.predict(
            x_test,
            batch_size=128,
            verbose=0
        )

        predicted_labels = np.argmax(
            predictions,
            axis=1
        )

        # Calculate classification metrics.
        test_accuracy = accuracy_score(
            y_test,
            predicted_labels
        )

        # Generate and save test confusion matrix.
        cm = confusion_matrix(
            y_test,
            predicted_labels,
            labels=np.arange(10)
        )

        cm_filename = (
            f"{name.lower()}_seed_{seed}_test_confusion_matrix.csv"
        )

        cm_path = os.path.join(
            confusion_matrix_dir,
            cm_filename
        )

        np.savetxt(
            cm_path,
            cm,
            delimiter=",",
            fmt="%d"
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test,
                predicted_labels,
                average="weighted",
                zero_division=0
            )
        )

        result = {
            "name": name,
            "seed": seed,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }

        all_results.append(result)

        print(f"\n{name} (seed={seed})")
        print(f"Test loss:      {test_loss:.4f}")
        print(f"Test accuracy:  {test_accuracy * 100:.3f}%")
        print(f"Test precision: {precision * 100:.3f}%")
        print(f"Test recall:    {recall * 100:.3f}%")
        print(f"Test F1:        {f1 * 100:.3f}%")


# --------------------------------------------------
# 5. Save individual test results
# --------------------------------------------------

run_path = "results/all_configurations_test_runs.csv"

fieldnames = [
    "name",
    "seed",
    "test_loss",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1"
]

with open(run_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_results)


# --------------------------------------------------
# 6. Calculate mean and standard deviation
# --------------------------------------------------

metrics = [
    "test_loss",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1"
]

summary_results = []

for name, _ in experiments:

    rows = [
        result for result in all_results
        if result["name"] == name
    ]

    summary = {"name": name}

    for metric in metrics:

        values = [
            result[metric] for result in rows
        ]

        summary[f"{metric}_mean"] = statistics.mean(values)

        summary[f"{metric}_std"] = statistics.stdev(values)

    summary_results.append(summary)


# --------------------------------------------------
# 7. Save summary results
# --------------------------------------------------

summary_path = (
    "results/all_configurations_test_summary.csv"
)

summary_fields = ["name"]

for metric in metrics:
    summary_fields.extend([
        f"{metric}_mean",
        f"{metric}_std"
    ])

with open(summary_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=summary_fields
    )

    writer.writeheader()
    writer.writerows(summary_results)


# --------------------------------------------------
# 8. Print final summary
# --------------------------------------------------

print("\n\nFINAL TEST PERFORMANCE")
print("=" * 90)

for result in summary_results:

    print(f"\n{result['name']}")

    for metric in metrics:

        mean = result[f"{metric}_mean"]
        std = result[f"{metric}_std"]

        if metric == "test_loss":
            print(f"{metric:20s}: {mean:.5f} ± {std:.5f}")
        else:
            print(
                f"{metric:20s}: "
                f"{mean * 100:.3f} ± {std * 100:.3f}%"
            )

print("\nAll experiments completed.")

print(f"Individual results: {run_path}")
print(f"Summary results:    {summary_path}")
