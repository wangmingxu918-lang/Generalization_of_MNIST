
import os
import re
import glob

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

seeds = [42, 123, 2025, 7, 99]

configurations = [
    ("baseline", "Baseline"),

    ("rotation_5_degrees", "Rotation ±5°"),
    ("rotation_10_degrees", "Rotation ±10°"),
    ("rotation_15_degrees", "Rotation ±15°"),

    ("translation_5_percent", "Translation 5%"),
    ("translation_10_percent", "Translation 10%"),
    ("translation_15_percent", "Translation 15%"),

    ("cropping_26x26", "Cropping 26 × 26"),
    ("cropping_24x24", "Cropping 24 × 24"),
    ("cropping_22x22", "Cropping 22 × 22"),

    ("flipping", "Horizontal flipping"),

    ("elastic_weak", "Elastic deformation\n(1.5 px)"),
    ("elastic_moderate", "Elastic deformation\n(3.0 px)"),

    ("gaussian_noise_0.05", "Gaussian noise (0.05)"),
    ("gaussian_noise_0.1", "Gaussian noise (0.10)"),
    ("gaussian_noise_0.15", "Gaussian noise (0.15)")
]

expected_configurations = {
    key for key, _ in configurations
}


# --------------------------------------------------
# 2. Find and validate test confusion matrices
# --------------------------------------------------

matrix_files = glob.glob(
    "results/**/*.csv",
    recursive=True
)

# Store one matrix per configuration and seed.
matrices = {}

for path in matrix_files:

    filename = os.path.basename(path).lower()

    # Extract the seed from the filename.
    match = re.search(
        r"_seed_(42|123|2025|7|99)(?:_|\.|$)",
        filename
    )

    if not match:
        continue

    seed = int(match.group(1))

    # Get configuration name from the filename.
    config = filename[:match.start()].rstrip("_")

    if config not in expected_configurations:
        continue

    try:
        cm = np.loadtxt(
            path,
            delimiter=",",
            dtype=int
        )
    except (ValueError, OSError):
        continue

    # Keep only valid 10 x 10 test matrices.
    if cm.shape != (10, 10):
        continue

    if cm.sum() != 10000:
        continue

    key = (config, seed)

    # Prefer the dedicated test_confusion_matrices directory
    # when duplicate copies exist.
    if key not in matrices:
        matrices[key] = (cm, path)

    elif "test_confusion_matrices" in path:
        matrices[key] = (cm, path)


# --------------------------------------------------
# 3. Check completeness
# --------------------------------------------------

missing = []

for config, _ in configurations:
    for seed in seeds:

        key = (config, seed)

        if key not in matrices:
            missing.append(key)

print("Unique matrices found:", len(matrices))
print("Expected matrices:", len(configurations) * len(seeds))

if missing:

    print("\nMissing matrices:")

    for config, seed in missing:
        print(f"{config}, seed={seed}")

    raise RuntimeError(
        "Some test confusion matrices are missing."
    )

print("\nAll 80 test confusion matrices are available.")





# --------------------------------------------------
# 4. Create one large 4 x 4 confusion matrix figure
# --------------------------------------------------

fig = plt.figure(figsize=(24, 24))

# Reserve the fifth column for a shared colour bar.
gs = fig.add_gridspec(
    4,
    5,
    width_ratios=[1, 1, 1, 1, 0.055],
    left=0.055,
    right=0.94,
    bottom=0.055,
    top=0.93,
    wspace=0.38,
    hspace=0.42
)

axes = []

for row in range(4):
    for col in range(4):
        axes.append(fig.add_subplot(gs[row, col]))

cbar_ax = fig.add_subplot(gs[:, 4])

image = None

# --------------------------------------------------
# 5. Plot all 16 configurations
# --------------------------------------------------

for index, (config, title) in enumerate(configurations):

    seed_matrices = [
        matrices[(config, seed)][0]
        for seed in seeds
    ]

    # Average raw confusion matrix counts across five seeds.
    mean_cm = np.mean(seed_matrices, axis=0)

    # Normalize each actual digit class into percentages.
    normalized_cm = (
        mean_cm /
        mean_cm.sum(axis=1, keepdims=True)
        * 100
    )

    ax = axes[index]

    image = ax.imshow(
        normalized_cm,
        cmap="Blues",
        vmin=0,
        vmax=100,
        interpolation="nearest"
    )

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=9
    )

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    ax.set_xticklabels(range(10), fontsize=8)
    ax.set_yticklabels(range(10), fontsize=8)

    ax.set_xlabel(
        "Predicted digit",
        fontsize=9,
        labelpad=4
    )

    ax.set_ylabel(
        "Actual digit",
        fontsize=9,
        labelpad=4
    )

    ax.tick_params(
        axis="both",
        length=2,
        pad=2
    )

    # Add numerical percentages to all 100 cells.
    for row in range(10):
        for col in range(10):

            value = normalized_cm[row, col]

            text_color = (
                "white" if value >= 50 else "black"
            )

            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=6,
                color=text_color
            )


# --------------------------------------------------
# 6. Shared colour bar
# --------------------------------------------------

colorbar = fig.colorbar(
    image,
    cax=cbar_ax
)

colorbar.set_label(
    "Classification percentage (%)",
    fontsize=11,
    labelpad=12
)

colorbar.ax.tick_params(labelsize=9)


# --------------------------------------------------
# 7. Figure title and output
# --------------------------------------------------

fig.suptitle(
    "Test Confusion Matrices Across CNN Data Augmentation Configurations",
    fontsize=17,
    fontweight="bold"
)

output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "figure_1_all_16_confusion_matrices.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nFigure saved successfully:")
print(output_path)
