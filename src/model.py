class FV_101(tf.keras.Model):
  def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int,
        activations: str,
        num_inputs: tuple[int],
        pool_size: tuple[int],
        units: int
        ):

    super(FV_101, self).__init__(name="")
    self.filters=filters
    self.kernel_size=kernel_size
    self.strides=strides
    self.activations = activations
    self.num_inputs=num_inputs
    self.pool_size=pool_size
    self.units=units

    self.conv2d_1 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=self.kernel_size,
        strides=self.strides,
        activation=self.activations,
        kernel_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
        bias_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
        input_shape=self.num_inputs,
        name="conv2d_1"
    )

    self.maxpool_1 = tf.keras.layers.MaxPool2D(
        pool_size=self.pool_size,
        strides=self.strides,
        padding="valid",
        name="maxpool_1"
    )


    self.conv2d_2 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=self.kernel_size,
        strides=self.strides,
        activation=self.activations,
        kernel_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
        bias_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
        name="conv2d_2"
    )

    self.maxpool_2 = tf.keras.layers.MaxPool2D(
        pool_size=self.pool_size,
        strides=self.strides,
        padding="valid",
        name="maxpool_2"
    )

    self.conv2d_3 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=self.kernel_size,
        strides=self.strides,
        activation=self.activations,
        kernel_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
        bias_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
        name="conv2d_3"
    )

    self.maxpool_3 = tf.keras.layers.MaxPool2D(
        pool_size=self.pool_size,
        strides=self.strides,
        padding="valid",
        name="maxpool_3"
    )

    self.conv2d_4 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=self.kernel_size,
        strides=self.strides,
        activation=self.activations,
        kernel_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
        bias_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
        name="conv2d_4"
    )

    self.maxpool_4 = tf.keras.layers.MaxPool2D(
        pool_size=self.pool_size,
        strides=self.strides,
        padding="valid",
        name="maxpool_4"
    )

    self.flat = tf.keras.layers.Flatten(name="flat_1")
    self.dns = tf.keras.layers.Dense(
        units=self.units,
        activation=self.activations,
        kernel_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
        bias_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
        name="dense_1"
        )
    self.outpt = tf.keras.layers.Dense(1, activation="sigmoid", name="output")


  def call(self, inputs):
      x = self.conv2d_1(inputs)
      x = self.maxpool_1(x)
      x = self.conv2d_2(x)
      x = self.maxpool_2(x)
      x = self.conv2d_3(x)
      x = self.maxpool_3(x)
      x = self.conv2d_4(x)
      x = self.maxpool_4(x)
      x = self.flat(x)
      x = self.dns(x)
      x = self.outpt(x)

      return x