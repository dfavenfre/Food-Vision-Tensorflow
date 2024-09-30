# Food-Vision-TensorFlow: Binary Classification (Steak vs Pizza)
This repository contains a custom Convolutional Neural Network (CNN) model built using TensorFlow for binary image classification on a subset of the Food-Vision dataset, specifically focusing on two classes: Steak and Pizza.

# Dataset
The dataset used in this project contains images of steak and pizza for binary classification. The dataset is split into training and test sets.
![image](https://github.com/user-attachments/assets/66aa2c56-c7bb-4e27-be46-bd7b1ee6b8a1)

# Image Preprocessing / Data Augmentation
To improve model generalization, data augmentation techniques such as rotation, zoom, width/height shifts, shear, and horizontal flipping are applied to the training dataset using the `ImageDataGenerator` class.

```Python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    directory="/content/pizza_steak/train",
    batch_size=32,
    target_size=(224,224),
    class_mode="binary",
    shuffle=False,
    seed=1212124
)

shuffled_training_data = train_datagen.flow_from_directory(
    directory="/content/pizza_steak/train",
    batch_size=32,
    target_size=(224,224),
    class_mode="binary",
    shuffle=True,
    seed=1212124
)

non_augmented_train_datagen = ImageDataGenerator(
    rescale=1./255
)

non_augmented_train_data = non_augmented_train_datagen.flow_from_directory(
    directory="/content/pizza_steak/train",
    batch_size=32,
    target_size=(224,224),
    class_mode="binary",
    shuffle=False,
    seed=1212124
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    directory="/content/pizza_steak/test",
    batch_size=32,
    target_size=(224,224),
    class_mode="binary",
    seed=1212124
)
```
### Augmented Data Sample
![image](https://github.com/user-attachments/assets/0697ccac-50a4-4402-ad45-aba52c29444a)

# Model Architecture
The CNN model, named FV_101, is built using several convolutional and max-pooling layers followed by a dense layer and a sigmoid output for binary classification.
```Python
class FV_101(tf.keras.Model):
    def __init__(
        self, filters: int, kernel_size: int, strides: int, activations: str,
        num_inputs: tuple[int], pool_size: tuple[int], units: int):
        super(FV_101, self).__init__(name="")
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activations = activations
        self.num_inputs = num_inputs
        self.pool_size = pool_size
        self.units = units

        self.conv2d_1 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
            activation=self.activations, kernel_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
            bias_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
            input_shape=self.num_inputs, name="conv2d_1"
        )

        self.maxpool_1 = tf.keras.layers.MaxPool2D(
            pool_size=self.pool_size, strides=self.strides, padding="valid", name="maxpool_1"
        )

        self.conv2d_2 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
            activation=self.activations, kernel_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
            bias_initializer=tf.keras.initializers.LecunNormal(seed=1212124), name="conv2d_2"
        )

        self.maxpool_2 = tf.keras.layers.MaxPool2D(
            pool_size=self.pool_size, strides=self.strides, padding="valid", name="maxpool_2"
        )

        self.conv2d_3 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
            activation=self.activations, kernel_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
            bias_initializer=tf.keras.initializers.LecunNormal(seed=1212124), name="conv2d_3"
        )

        self.maxpool_3 = tf.keras.layers.MaxPool2D(
            pool_size=self.pool_size, strides=self.strides, padding="valid", name="maxpool_3"
        )

        self.conv2d_4 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
            activation=self.activations, kernel_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
            bias_initializer=tf.keras.initializers.LecunNormal(seed=1212124), name="conv2d_4"
        )

        self.maxpool_4 = tf.keras.layers.MaxPool2D(
            pool_size=self.pool_size, strides=self.strides, padding="valid", name="maxpool_4"
        )

        self.flat = tf.keras.layers.Flatten(name="flat_1")
        self.dns = tf.keras.layers.Dense(
            units=self.units, activation=self.activations,
            kernel_initializer=tf.keras.initializers.LecunNormal(seed=1212124),
            bias_initializer=tf.keras.initializers.LecunNormal(seed=1212124), name="dense_1"
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
```


# Model Compilation
The model is compiled with the `binary_crossentropy` loss function and the `Adam` optimizer, tracking the `accuracy` metric.

```Python
cnn_model = FV_101(
    filters=10,
    kernel_size=3,
    strides=2,
    activations="relu",
    num_inputs=(224, 224, 3),
    pool_size=2,
    units=100
)

cnn_model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)
```
# Model Fitting
The model is trained for 100 epochs with data augmentation and checkpoint callbacks to save the best model.

```Python
cnn_model.fit(
    shuffled_training_data,
    epochs=100,
    steps_per_epoch=len(train_data),
    batch_size=32,
    callbacks=[
        cb_checkpoint,  # Checkpoint saving
        cb_reducelr,    # Reduce learning rate on plateau
        cb_earlystop,   # Early stopping
        tensorboard_callback  # TensorBoard logging
    ],
    verbose=1,
    validation_data=test_data,
    validation_steps=len(test_data)
)
```

# Evaluation
After training, the model is evaluated on the test dataset to measure its performance. To see detailed 
results follow the [Wandb link](https://wandb.ai/alone-wolf/Food-Vision%20Model/reports/Food-Vision-CNN-Binary-Model-Development--Vmlldzo5NTUxMzYx?accessToken=t76o08f5sbzrow4xptstvlheara0abb8q7nsksh2ty6urhalwmvyax1o1ouknk66) 
![image](https://github.com/user-attachments/assets/8b7dd1c8-edeb-491b-a48f-61c885659935)

## Validation Accuracy
![W B Chart 9_30_2024, 2_35_04 PM](https://github.com/user-attachments/assets/8499ae1e-7589-4a30-9858-7f21f2bc4d91)

## Training/Validation Loss
![W B Chart 9_30_2024, 2_35_31 PM](https://github.com/user-attachments/assets/830d0ab7-32d1-4a9f-8238-c7228dd723ac)

## Hyperparameter Importance
![W B Chart 9_30_2024, 2_35_42 PM](https://github.com/user-attachments/assets/9a668bf9-f154-4af8-8df1-e471bd2786a8)


# Requirements
To run this project, you will need to install the following dependencies:

* Python 3.x
* TensorFlow
* NumPy
* Matplotlib (for plotting)

# Usage
```bash
git clone https://github.com/your-repo/food-vision-tensorflow.git
pip install -r requirements.txt
```
