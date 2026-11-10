
import csv
import os
import statistics

import numpy as np
import tensorflow as tf

from data_preprocessing import (
    x_train,
    y_train,
    x_val,
    y_val
)

from model import build_model
from evaluation import evaluate_model


def run_experiment(name, augmentation_factory=None, seed=42):

    # Reset TensorFlow state between experiments.
    tf.keras.backend.clear_session()

    # Set the seed before creating the augmentation layer and model.
    tf.keras.utils.set_random_seed(seed)

    augmentation = (
        augmentation_factory()
        if augmentation_factory is not None
        else None
    )

    model = build_model(augmentation=augmentation)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=128,
        verbose=1
    )

    # Evaluate on the validation set, NOT the test set.
    results = evaluate_model(model, x_val, y_val)

    experiment_results = {
        "name": name,
        "seed": seed,
        "train_accuracy": history.history["accuracy"][-1],
        "val_loss": results["loss"],
        "val_accuracy": results["accuracy"],
        "val_precision": results["precision"],
        "val_recall": results["recall"],
        "val_f1": results["f1"],
        "confusion_matrix": results["confusion_matrix"]
    }

    print(f"\n{name} Results (seed={seed})")
    print("---------------------")

    for metric in [
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1"
    ]:
        print(f"{metric:20s}: {experiment_results[metric]:.4f}")

    print("\nValidation Confusion Matrix:")
    print(experiment_results["confusion_matrix"])

    return experiment_results


def run_multiple_seeds(name, augmentation_factory, seeds):

    all_results = []

    os.makedirs("results", exist_ok=True)

    for seed in seeds:

        print(f"\n\nRunning {name} with seed {seed}")
        print("=" * 50)

        results = run_experiment(
            name=name,
            augmentation_factory=augmentation_factory,
            seed=seed
        )

        all_results.append(results)

        # Save this run's validation confusion matrix.
        confusion_path = (
            f"results/{name}_seed_{seed}_validation_confusion_matrix.csv"
        )

        np.savetxt(
            confusion_path,
            results["confusion_matrix"],
            delimiter=",",
            fmt="%d"
        )

    # Metrics to summarize across runs.
    metric_names = [
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1"
    ]

    summary = {}

    for metric in metric_names:

        values = [result[metric] for result in all_results]

        summary[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0
        }

    # Save individual run results.
    run_path = f"results/{name}_runs.csv"

    with open(run_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["name", "seed"] + metric_names
        )

        writer.writeheader()

        for result in all_results:

            row = {
                "name": result["name"],
                "seed": result["seed"]
            }

            for metric in metric_names:
                row[metric] = result[metric]

            writer.writerow(row)

    # Save mean and standard deviation.
    summary_path = f"results/{name}_summary.csv"

    with open(summary_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["Metric", "Mean", "Standard Deviation"])

        for metric in metric_names:

            writer.writerow([
                metric,
                summary[metric]["mean"],
                summary[metric]["std"]
            ])

    print(f"\n{name} Summary Across {len(seeds)} Seeds")
    print("=" * 50)

    for metric in metric_names:

        mean = summary[metric]["mean"]
        std = summary[metric]["std"]

        print(f"{metric:20s}: {mean:.4f} ± {std:.4f}")

    print(f"\nIndividual results saved to: {run_path}")
    print(f"Summary saved to: {summary_path}")

    return all_results, summary
