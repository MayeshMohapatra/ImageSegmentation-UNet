import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf


def read_image(dir):
    x = cv2.imread(dir, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(dir):
    x = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def load_data(dir):
    images = glob(os.path.join(dir, "images/*"))
    masks = glob(os.path.join(dir, "masks/*"))

    train_x, test_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, test_y = train_test_split(masks, test_size=0.2, random_state=42)

    return (train_x, train_y), (test_x, test_y)


def preprocessing(image_path, mask_path):
    def func(image_path, mask_path):
        image_path = image_path.decode()
        mask_path = mask_path.decode()

        x = read_image(image_path)
        y = read_mask(mask_path)

        return x, y

    image, mask = tf.numpy_function(
        func, [image_path, mask_path], [tf.float32, tf.float32]
    )
    image.set_shape([256, 256, 3])
    mask.set_shape([256, 256, 1])

    return image, mask


def tf_dataset(images, masks, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocessing)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
    dataset_path = "people_segmentation"

    (train_x, train_y), (test_x, test_y) = load_data(dataset_path)
    print(f"Training: {len(train_x)} - {len(train_y)}")
    print(f"Testing: {len(test_x)} - {len(test_y)}")
    # print(f"Train: {len(train_x)} - {len(train_y)}")
    # print(f"Test: {len(test_x)} - {len(test_y)}")
    y = read_mask(train_y[1])
    cv2.imwrite("2.png", y * 255)
    train_dataset = tf_dataset(train_x, train_y, batch=8)
    batch = 12
    train_steps = len(train_x) // batch

    if len(train_x) % batch != 0:
        train_steps += 1

    print(f"Train steps: {train_steps}")
