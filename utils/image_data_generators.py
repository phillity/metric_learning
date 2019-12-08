import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def sparse_to_onehot(labels, num_classes):
    targets = np.array(labels).reshape(-1)
    return np.eye(num_classes)[targets.astype(int)]


class CustomImageDataGenerator:
    def __init__(self, featurewise_center=False, samplewise_center=False,
                 featurewise_std_normalization=False, samplewise_std_normalization=False,
                 zca_epsilon=1e-06, zca_whitening=False, rotation_range=0,
                 width_shift_range=0.0, height_shift_range=0.0, brightness_range=None,
                 shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0,
                 fill_mode="nearest", cval=0.0, horizontal_flip=False, vertical_flip=False,
                 rescale=None, preprocessing_function=None, data_format="channels_last",
                 validation_split=0.0, dtype="float32"):

        self.datagen = ImageDataGenerator(
            featurewise_center=featurewise_center, samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening, zca_epsilon=zca_epsilon, rotation_range=rotation_range,
            width_shift_range=width_shift_range, height_shift_range=height_shift_range,
            brightness_range=brightness_range, shear_range=shear_range, zoom_range=zoom_range,
            channel_shift_range=channel_shift_range, fill_mode=fill_mode, cval=cval,
            horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
            rescale=rescale, preprocessing_function=preprocessing_function, data_format=data_format,
            validation_split=validation_split, dtype=dtype)

    def flow_from_directory(self, directory, target_size=(256, 256), color_mode="rgb",
                            classes=None, class_mode="categorical", batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix="", save_format="png", follow_links=False,
                            subset=None, interpolation="nearest"):

        self.generator = self.datagen.flow_from_directory(
            directory, target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode, batch_size=batch_size,
            shuffle=shuffle, seed=seed, save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format, follow_links=follow_links, subset=subset, interpolation=interpolation)

        self.num_classes = self.generator.num_classes

    def flow(self):
        while True:
            x_batch, y_batch = next(self.generator)
            y_onehot_batch = sparse_to_onehot(
                y_batch, self.generator.num_classes)
            yield [x_batch, y_onehot_batch], y_batch


def get_generators(dataset_name, batch_size, target_size, metric):
    dataset_path = os.path.abspath(os.path.join(os.path.abspath(
        ""), "data", dataset_name + "_dataset"))

    if metric == "lmcl" or metric == "aaml":
        train_generator = CustomImageDataGenerator(
            rescale=1. / 255, fill_mode="nearest",
            shear_range=0.05, zoom_range=0.05,
            width_shift_range=0.05, height_shift_range=0.05,
            rotation_range=10.0, horizontal_flip=True)
        val_generator = CustomImageDataGenerator(rescale=1. / 255)
        test_generator = CustomImageDataGenerator(rescale=1. / 255)

        train_generator.flow_from_directory(
            os.path.join(dataset_path, "train"), shuffle=True, seed=42,
            batch_size=batch_size, target_size=(target_size, target_size),
            class_mode="sparse")
        val_generator.flow_from_directory(
            os.path.join(dataset_path, "val"), shuffle=True,
            seed=42, batch_size=batch_size, target_size=(target_size, target_size),
            class_mode="sparse")
        test_generator.flow_from_directory(
            os.path.join(dataset_path, "test"), shuffle=True,
            seed=42, batch_size=batch_size, target_size=(target_size, target_size),
            class_mode="sparse")
    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255, fill_mode="nearest",
            shear_range=0.05, zoom_range=0.05,
            width_shift_range=0.05, height_shift_range=0.05,
            rotation_range=10.0, horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            os.path.join(dataset_path, "train"), shuffle=True, seed=42,
            batch_size=batch_size, target_size=(target_size, target_size),
            class_mode="sparse")
        val_generator = val_datagen.flow_from_directory(
            os.path.join(dataset_path, "val"), shuffle=True,
            seed=42, batch_size=batch_size, target_size=(target_size, target_size),
            class_mode="sparse")
        test_generator = test_datagen.flow_from_directory(
            os.path.join(dataset_path, "test"), shuffle=True,
            seed=42, batch_size=batch_size, target_size=(target_size, target_size),
            class_mode="sparse")
    return [train_generator, val_generator, test_generator]
