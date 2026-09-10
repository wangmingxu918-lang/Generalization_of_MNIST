import tensorflow as tf


def build_model(augmentation=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1))
    ])

    if augmentation is not None:
        model.add(augmentation)

    model.add(tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ))

    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ))

    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(
        128, activation="relu"
    ))

    model.add(tf.keras.layers.Dense(
        10, activation="softmax"
    ))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model