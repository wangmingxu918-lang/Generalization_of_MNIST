
import matplotlib.pyplot as plt
import tensorflow as tf

from data_preprocessing import x_train
from augmentations.elastic import ElasticDeformation


sample_images = tf.convert_to_tensor(x_train[:8])

weak_layer = ElasticDeformation(
    max_displacement=1.5
)

moderate_layer = ElasticDeformation(
    max_displacement=3.0
)

weak_images = weak_layer(
    sample_images,
    training=True
).numpy()

moderate_images = moderate_layer(
    sample_images,
    training=True
).numpy()


fig, axes = plt.subplots(3, 8, figsize=(12, 5))

for i in range(8):

    axes[0, i].imshow(
        sample_images[i, :, :, 0],
        cmap="gray"
    )

    axes[1, i].imshow(
        weak_images[i, :, :, 0],
        cmap="gray"
    )

    axes[2, i].imshow(
        moderate_images[i, :, :, 0],
        cmap="gray"
    )

    for row in range(3):
        axes[row, i].axis("off")

axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Weak")
axes[2, 0].set_ylabel("Moderate")

plt.tight_layout()
plt.show()
