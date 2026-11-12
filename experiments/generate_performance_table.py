
import csv
import statistics
from pathlib import Path

RESULTS_DIR = Path("results")

# The 16 configurations from the completed experiments.
experiments = {
    "Baseline": "Baseline_runs.csv",
    "Rotation ±10°": "Rotation_runs.csv",
    "Rotation ±5°": "Rotation_5_degrees_runs.csv",
    "Rotation ±15°": "Rotation_15_degrees_runs.csv",
    "Translation 10%": "Translation_runs.csv",
    "Translation 5%": "Translation_5_percent_runs.csv",
    "Translation 15%": "Translation_15_percent_runs.csv",
    "Cropping 24 × 24": "Cropping_24x24_runs.csv",
    "Cropping 26 × 26": "Cropping_26x26_runs.csv",
    "Cropping 22 × 22": "Cropping_22x22_runs.csv",
    "Horizontal flipping": "Flipping_runs.csv",
    "Elastic deformation (1.5 px)": "Elastic_Weak_runs.csv",
    "Elastic deformation (3.0 px)": "Elastic_Moderate_runs.csv",
    "Gaussian noise (0.05)": "Gaussian_Noise_0.05_runs.csv",
    "Gaussian noise (0.10)": "Gaussian_Noise_0.1_runs.csv",
    "Gaussian noise (0.15)": "Gaussian_Noise_0.15_runs.csv",
}


def calculate_mean_std(values):
    if len(values) != 5:
        raise ValueError(
            f"Expected 5 runs, but found {len(values)}."
        )

    mean = statistics.mean(values) * 100
    std = statistics.stdev(values) * 100

    return f"{mean:.3f} ± {std:.3f}"


def read_runs(filename):
    path = RESULTS_DIR / filename

    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 5:
        raise ValueError(
            f"{filename}: expected 5 rows, found {len(rows)}"
        )

    return rows


def get_test_accuracy(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        return "—"

    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 5:
        raise ValueError(
            f"{filename}: expected 5 test runs, found {len(rows)}"
        )

    values = [float(row["test_accuracy"]) for row in rows]

    return calculate_mean_std(values)


table = []

for name, filename in experiments.items():
    rows = read_runs(filename)

    training = [
        float(row["train_accuracy"]) for row in rows
    ]

    validation = [
        float(row["val_accuracy"]) for row in rows
    ]

    precision = [
        float(row["val_precision"]) for row in rows
    ]

    recall = [
        float(row["val_recall"]) for row in rows
    ]

    # Test accuracy is available only for the two
    # configurations evaluated on the held-out test set.
    test_filename = {
        "Baseline": "Baseline_test_runs.csv",
        "Elastic deformation (1.5 px)": "Elastic_Weak_test_runs.csv"
    }.get(name)

    test_accuracy = (
        get_test_accuracy(test_filename)
        if test_filename else "—"
    )

    table.append({
        "Configuration": name,
        "Validation Acc. (%)": calculate_mean_std(validation),
        "Training Acc. (%)": calculate_mean_std(training),
        "Test Acc. (%)": test_accuracy,
        "Validation Precision (%)": calculate_mean_std(precision),
        "Validation Recall (%)": calculate_mean_std(recall)
    })


# Print the table
headers = list(table[0].keys())

print("\nCLASSIFICATION PERFORMANCE ACROSS EXPERIMENTAL GROUPS\n")

print(" | ".join(headers))
print("-" * 150)

for row in table:
    print(" | ".join(row[h] for h in headers))


# Save the table as a CSV file
output_path = RESULTS_DIR / "classification_performance_all_groups.csv"

with open(output_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    writer.writerows(table)

print(f"\nTable saved to: {output_path}")
