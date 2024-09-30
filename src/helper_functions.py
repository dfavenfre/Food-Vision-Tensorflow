def view_random_images(
    target_dir,
    target_class
    ):
  target_folder = os.path.join(target_dir, target_class)
  random_image = random.sample(os.listdir(target_folder), 1)
  img = mpimg.imread(target_folder + "/" + random_image[0])

  plt.imshow(img)
  plt.title(target_class + " with a shape of: " + str(img.shape))
  plt.axis("off")

  return img

def resize_images(
    file_path: str,
    img_shape: int = 224
    ):
  image = tf.io.read_file(file_path)
  image = tf.image.decode_image(image)
  image = tf.image.resize(image,
                          size=[img_shape, img_shape]
                          )
  image = tf.expand_dims(image / 255., axis=0)

  return image

def plot_predictions(
    target_dir: str,
    model
    ):
  class_labels = ["pizza", "steak"]
  target_class = random.choice(class_labels)
  target_folder = os.path.join(target_dir, target_class)

  random_image = random.choice(os.listdir(target_folder))
  image_path = os.path.join(target_folder,
                            random_image)
  resized_image = resize_images(image_path)

  prediction = model.predict(resized_image)
  prediction_object = class_labels[int(tf.round(prediction))]

  image = mpimg.imread(image_path)
  plt.imshow(image)
  plt.title(f"Predicted Object: {prediction_object}")
  plt.axis(False)
  plt.show()