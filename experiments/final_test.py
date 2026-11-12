
import os
import csv
import statistics
import numpy as np
import tensorflow as tf

from data_preprocessing import (
    x_train, y_train,
    x_val, y_val,
    x_test, y_test
)
from model import build_model
from evaluation import evaluate_model
from augmentations.elastic import ElasticDeformation


# Five random seeds used in the original experiments
seeds = [42, 123, 2025, 7, 99]

# Combine the original training and validation datasets
x_train_full = np.concatenate([x_train, x_val], axis=0)
y_train_full = np.concatenate([y_train, y_val], axis=0)

print("Final training dataset:", x_train_full.shape)
print("Final test dataset:", x_test.shape)

os.makedirs("results", exist_ok=True)


def run_final_evaluation(name, augmentation_factory):
    all_results = []
    total_confusion_matrix = np.zeros((10, 10), dtype=int)

    for seed in seeds:
        print(f"\n{name} - Seed {seed}")

        # Reset randomness before building each model
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        augmentation = (
            augmentation_factory()
            if augmentation_factory is not None
            else None
        )

        model = build_model(augmentation)

        # Train using all 60,000 training images
        model.fit(
            x_train_full,
            y_train_full,
            epochs=10,
            batch_size=128,
            verbose=2
        )

        # Evaluate on the untouched test set
        result = evaluate_model(model, x_test, y_test)

        row = {
            "name": name,
            "seed": seed,
            "test_loss": result["loss"],
            "test_accuracy": result["accuracy"],
            "test_precision": result["precision"],
            "test_recall": result["recall"],
            "test_f1": result["f1"]
        }

        all_results.append(row)

        total_confusion_matrix += result["confusion_matrix"]

        print(
            f"Test accuracy: "
            f"{result['accuracy'] * 100:.4f}%"
        )

        # Save this run's confusion matrix
        np.savetxt(
            f"results/{name}_seed_{seed}_test_confusion_matrix.csv",
            result["confusion_matrix"],
            delimiter=",",
            fmt="%d"
        )

    # Save individual run results
    with open(f"results/{name}_test_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(all_results)

    # Calculate mean and standard deviation
    metrics = [
        "test_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1"
    ]

    summary = {}

    for metric in metrics:
        values = [row[metric] for row in all_results]
        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_std"] = statistics.stdev(values)

    with open(f"results/{name}_test_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary.keys()
        )
        writer.writeheader()
        writer.writerow(summary)

    # Save the combined confusion matrix across five runs
    np.savetxt(
        f"results/{name}_combined_test_confusion_matrix.csv",
        total_confusion_matrix,
        delimiter=",",
        fmt="%d"
    )

    print(f"\n{name} summary:")
    for metric in metrics:
        print(
            f"{metric}: "
            f"{summary[f'{metric}_mean']:.5f} "
            f"+/- {summary[f'{metric}_std']:.5f}"
        )


# 1. Baseline
run_final_evaluation(
    name="Baseline",
    augmentation_factory=None
)

# 2. Weak elastic deformation
def weak_elastic_augmentation():
    return ElasticDeformation(
        max_displacement=1.5,
        grid_size=4
    )

run_final_evaluation(
    name="Elastic_Weak",
    augmentation_factory=weak_elastic_augmentation
)

print("\nFinal test evaluation completed!")
