import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from utils.image_data_generators import get_generators
from losses.angular_margin_loss.cosface_loss import CosFace
from losses.angular_margin_loss.arcface_loss import ArcFace


np.random.seed(42)
tf.random.set_seed(42)
tf.keras.backend.clear_session()


def get_model(pretrained_model, num_classes, embedding_size, target_size, metric):
    onehot_labels = L.Input(shape=(num_classes,))
    # get feature vector extracted using DenseNet
    feature_extractor = Model(inputs=pretrained_model.input,
                              outputs=model.get_layer("flatten").output)
    x = L.Flatten()(feat_extractor.output)
    # BN-Dropout-FC-BN
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.25)(x)
    x = L.Dense(embedding_size, activation="relu")(x)
    x = L.BatchNormalization()(x)
    # make embeddings unit length
    embeddings = L.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)
    # compile model with metric learning loss function
    if metric == "contrastive":
        model = Model(inputs=feat_extractor.input, outputs=embeddings)
        model.compile("adam", loss=contrastive_loss)
    elif metric == "triplet":
        model = Model(inputs=feat_extractor.input, outputs=embeddings)
        model.compile("adam", loss=semihard_triplet_loss)
    elif metric == "lmcl":
        predictions = CosFace(num_classes=num_classes)(
            [embeddings, onehot_labels])
        model = Model(inputs=[feat_extractor.input,
                              onehot_labels], outputs=predictions)
        model.compile(
            "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    elif metric == "aaml":
        predictions = ArcFace(num_classes=num_classes)(
            [embeddings, onehot_labels])
        model = Model(inputs=feat_extractor.input, outputs=embeddings)
        model.compile(
            "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:
        predictions = L.Dense(num_classes, activation="softmax")(embeddings)
        model = Model(inputs=feat_extractor.input, outputs=predictions)
        model.compile(
            "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(finetune_model, dataset_name, generators, embedding_size, metric=None):
    train_generator, val_generator = generators[0], generators[1]

    if metric == "contrastive":
        out_path = os.path.join(os.path.abspath(
            ""), "models", dataset_name + "_finetuned_contrastive_" + str(embedding_size) + "d.h5")
    elif metric == "triplet":
        out_path = os.path.join(os.path.abspath(
            ""), "models", dataset_name + "_finetuned_triplet_" + str(embedding_size) + "d.h5")
    elif metric == "lmcl":
        out_path = os.path.join(os.path.abspath(
            ""), "models", dataset_name + "_finetuned_lmcl_" + str(embedding_size) + "d.h5")
    elif metric == "aaml":
        out_path = os.path.join(os.path.abspath(
            ""), "models", dataset_name + "_finetuned_aaml_" + str(embedding_size) + "d.h5")
    else:
        out_path = os.path.join(os.path.abspath(
            ""), "models", dataset_name + "_finetuned_softmax_" + str(embedding_size) + "d.h5")
    log = CSVLogger(out_path[:-2] + "log")
    mc = ModelCheckpoint(out_path, monitor="val_loss",
                         save_best_only=True, verbose=1)
    es = EarlyStopping(patience=3, monitor="val_loss",
                       restore_best_weights=True, verbose=1)

    if metric == "lmcl" or metric == "aaml":
        history = finetune_model.fit_generator(
            train_generator.flow(),
            steps_per_epoch=train_generator.generator.n // train_generator.generator.batch_size,
            validation_data=val_generator.flow(),
            validation_steps=val_generator.generator.n // val_generator.generator.batch_size,
            shuffle=False, epochs=1000, callbacks=[mc, es, log], verbose=1)
    else:
        history = finetune_model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.n // train_generator.batch_size,
            validation_data=val_generator,
            validation_steps=val_generator.n // val_generator.batch_size,
            shuffle=False, epochs=1000, callbacks=[mc, es, log], verbose=1)

    return finetune_model, history


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--metric", required=False, default="softmax", choices=["softmax", "contrastive", "triplet", "lmcl", "aaml"],
                        help="loss metric to use in training")
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=64,
                        help="batch size to use in training")
    parser.add_argument("-e", "--embedding_size", required=False, type=int, default=512,
                        help="embedding size to use in training")
    parser.add_argument("-i", "--image_size", required=False, type=int, default=96,
                        help="image size to use in training")
    args = vars(parser.parse_args())

    dataset_name = "test"

    pretrained_model = load_model(os.path.join(os.path.abspath(
        ""), "models", "train_pretrained_softmax_" + str(embedding_size) + "d.h5"))
    # print(pretrained_model.summary())

    generators = get_generators(
        dataset_name, batch_size=args["batch_size"], target_size=args["image_size"], metric=args["metric"])
    train_generator, val_generator, test_generator = generators[0], generators[1], generators[2]

    finetune_model = get_model(pretrained_model, num_classes=train_generator.num_classes,
                               embedding_size=args["embedding_size"], target_size=args["image_size"], metric=args["metric"])

    model = train_model(
        finetune_model, dataset_name, generators, embedding_size=args["embedding_size"], metric=args["metric"])
