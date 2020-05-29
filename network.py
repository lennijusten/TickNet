# Documentation
# https://www.tensorflow.org/tutorials/load_data/images

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE

main_dir = '/Users/Lenni/Documents/Independent Research Project/Data/test_train/Test Data'
data_dir = pathlib.Path(main_dir)
# data_set = tf.keras.preprocessing.image_dataset_from_directory(main_dir, labels='inferred')

list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=None)
image_count = len(list(data_dir.glob('*/*.JPG')))
print(image_count)

class_names = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])
print(class_names)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    # returns true for the corresponding class name
    return parts[-2] == class_names


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


# for image, label in labeled_ds.take(1):
#     print("Image shape: ", image.numpy().shape)
#     print("Label: ", label.numpy())


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


default_timeit_steps = 1000


def timeit(ds, steps=default_timeit_steps):
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        batch = next(it)
        if i % 10 == 0:
            print('.', end='')
    print()
    end = time.time()

    duration = end - start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(BATCH_SIZE * steps / duration))


# 1000 batches: 58.49s, 547.1 Images/s
# train_ds = prepare_for_training(labeled_ds)
# timeit(train_ds)

# 1000 batches: 30.05s, 1064.9 Images/s
filecache_ds = prepare_for_training(labeled_ds, cache="./Test Data.tfcache")
timeit(filecache_ds)

val_dataset = filecache_ds.take(417)
train_dataset = filecache_ds.skip(417)

# image_batch, label_batch = next(iter(filecache_ds))
sample_images, _ = next(iter(val_dataset))


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plotImages(sample_images[:5])

# model = Sequential([
#     Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
#     MaxPooling2D(),
#     Conv2D(32, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Conv2D(64, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(1)
# ])
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])

IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
# Pre-trained model with MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Freeze the pre-trained model weights
base_model.trainable = False

# Trainable classification head
maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(4, activation='sigmoid')

# Layer classification head with feature detector
model = tf.keras.Sequential([
    base_model,
    maxpool_layer,
    prediction_layer
])

learning_rate = 0.0005

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy']
)

model.summary()

num_epochs = 30
steps_per_epoch = round(image_count-417)//BATCH_SIZE
val_steps = 20
model.fit(train_dataset.repeat(),
          epochs=num_epochs,
          steps_per_epoch = steps_per_epoch,
          validation_data=val_dataset.repeat(),
          validation_steps=val_steps)