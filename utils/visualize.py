import matplotlib.pyplot as plt
from softmax_loss.softmax_model import get_generators, get_softmax_model, train_softmax_model


def visualize(model_embeddings, generators, title):
    train_generator = generators[0]
    y_train = test_generator.labels
    emb_train = model_embeddings.predict_generator(test_generator, verbose=1)

    labels = np.unique(y_train)
    for label in labels:
        emb = emb_train[np.where(y_train == label)[0], :]
        lab = y_train[np.where(y_train == label)[0]]
        plt.plot(emb, lab)

    plt.title(title)
    plt.show()
    plt.savefig("_".join(title.split()) + ".png")


if __name__ == "__main__":
    dataset_name = "gtfd"
    batch_size = 16
    embedding_size = 2
    target_size = (96, 96)

    generators = get_generators(dataset_name, batch_size=batch_size, target_size=target_size)
    train_generator, val_generator, test_generator = generators[0], generators[1], generators[2]

    softmax_model = get_softmax_model(num_classes=train_generator.num_classes, embedding_size=embedding_size, target_size=target_size)
    softmax_embeddings = Model(inputs=softmax_model.input, outputs=softmax_model.get_layer("lambda").output)

    visualize(softmax_embeddings, generators, "Softmax-Learned Embeddings")
