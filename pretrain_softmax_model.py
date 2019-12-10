import os
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.applications.densenet import DenseNet121
from utils.image_data_generators import get_generators


np.random.seed(42)
tf.random.set_seed(42)
tf.keras.backend.clear_session()


def get_model(num_classes, embedding_size, target_size):
    # get feature vector extracted using DenseNet
    feat_extractor = DenseNet121(
        input_shape=(target_size, target_size, 3),
        include_top=False, weights=None)
    x = L.Flatten()(feat_extractor.output)
    # BN-Dropout-FC-BN
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.25)(x)
    x = L.Dense(embedding_size, activation="relu")(x)
    x = L.BatchNormalization()(x)
    # compile model with softmax loss function
    predictions = L.Dense(num_classes, activation="softmax")(embeddings)
    model = Model(inputs=feat_extractor.input, outputs=predictions)
    model.compile("adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_model(model, dataset_name, generators, embedding_size):
    train_generator, val_generator = generators[0], generators[1]

    out_path = os.path.join(os.path.abspath(
        ""), "models", dataset_name + "_pretrained_softmax_" + str(embedding_size) + "d.h5")
    log = CSVLogger(out_path[:-2] + "log")

    mc = ModelCheckpoint(out_path, monitor="val_loss",
                         save_best_only=True, verbose=1)
    es = EarlyStopping(patience=5, monitor="val_loss",
                       restore_best_weights=True, verbose=1)
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.n // train_generator.batch_size,
                                  validation_data=val_generator,
                                  validation_steps=val_generator.n // val_generator.batch_size,
                                  shuffle=False, epochs=1000, callbacks=[log, mc, es], verbose=1)
    return model, history


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=64,
                        help="batch size to use in training")
    parser.add_argument("-e", "--embedding_size", required=False, type=int, default=512,
                        help="embedding size to use in training")
    parser.add_argument("-i", "--image_size", required=False, type=int, default=96,
                        help="image size to use in training")
    args = vars(parser.parse_args())

    dataset = "train"

    generators = get_generators(
        dataset, batch_size=args["batch_size"], target_size=args["image_size"])
    train_generator, val_generator, test_generator = generators[0], generators[1], generators[2]

    model = get_model(num_classes=train_generator.num_classes,
                      embedding_size=args["embedding_size"], target_size=args["image_size"])

    model = train_model(
        model, dataset, generators, embedding_size=args["embedding_size"])
