
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="MNISTAugmentation"
)
class ElasticDeformation(tf.keras.layers.Layer):

    def __init__(
        self,
        max_displacement=1.5,
        grid_size=4,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.max_displacement = max_displacement
        self.grid_size = grid_size

    def call(self, images, training=None):

        if training is None:
            training = False

        if isinstance(training, bool):
            if training:
                return self._deform(images)
            return images

        return tf.cond(
            tf.cast(training, tf.bool),
            lambda: self._deform(images),
            lambda: tf.identity(images)
        )

    def _deform(self, images):

        images = tf.cast(images, tf.float32)

        shape = tf.shape(images)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        # Generate a small random displacement field.
        # The last dimension stores vertical and horizontal shifts.
        displacement = tf.random.uniform(
            shape=[
                batch_size,
                self.grid_size,
                self.grid_size,
                2
            ],
            minval=-self.max_displacement,
            maxval=self.max_displacement,
            dtype=tf.float32
        )

        # Smoothly enlarge the displacement field to image size.
        displacement = tf.image.resize(
            displacement,
            size=[height, width],
            method="bicubic"
        )

        # Create the original pixel coordinate grid.
        y_grid, x_grid = tf.meshgrid(
            tf.range(height, dtype=tf.float32),
            tf.range(width, dtype=tf.float32),
            indexing="ij"
        )

        # Calculate the source coordinates for each output pixel.
        source_y = y_grid[tf.newaxis, :, :] + displacement[..., 0]
        source_x = x_grid[tf.newaxis, :, :] + displacement[..., 1]

        # Find the four neighbouring pixels.
        y0 = tf.floor(source_y)
        x0 = tf.floor(source_x)

        y1 = y0 + 1
        x1 = x0 + 1

        # Calculate bilinear interpolation weights.
        wa = (y1 - source_y) * (x1 - source_x)
        wb = (y1 - source_y) * (source_x - x0)
        wc = (source_y - y0) * (x1 - source_x)
        wd = (source_y - y0) * (source_x - x0)

        # Convert coordinates to integer indices.
        y0 = tf.cast(y0, tf.int32)
        x0 = tf.cast(x0, tf.int32)
        y1 = tf.cast(y1, tf.int32)
        x1 = tf.cast(x1, tf.int32)

        # Build batch indices for gather_nd.
        batch_indices = tf.broadcast_to(
            tf.range(batch_size)[:, tf.newaxis, tf.newaxis],
            [batch_size, height, width]
        )

        def get_pixels(y, x):

            # Identify coordinates outside the image.
            valid = (
                (y >= 0) & (y < height) &
                (x >= 0) & (x < width)
            )

            # Clip indices so gather_nd remains valid.
            y = tf.clip_by_value(y, 0, height - 1)
            x = tf.clip_by_value(x, 0, width - 1)

            indices = tf.stack(
                [batch_indices, y, x],
                axis=-1
            )

            pixels = tf.gather_nd(images, indices)

            # Use zero padding outside the image.
            valid = tf.cast(valid[..., tf.newaxis], tf.float32)

            return pixels * valid

        # Retrieve the four neighbouring pixel values.
        Ia = get_pixels(y0, x0)
        Ib = get_pixels(y0, x1)
        Ic = get_pixels(y1, x0)
        Id = get_pixels(y1, x1)

        # Bilinear interpolation.
        output = (
            wa[..., tf.newaxis] * Ia +
            wb[..., tf.newaxis] * Ib +
            wc[..., tf.newaxis] * Ic +
            wd[..., tf.newaxis] * Id
        )

        return output

    def get_config(self):

        config = super().get_config()

        config.update({
            "max_displacement": self.max_displacement,
            "grid_size": self.grid_size
        })

        return config
