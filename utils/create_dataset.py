import os
import cv2
import stat
import shutil
from sklearn.model_selection import train_test_split
from preprocess import align_image


__all__ = [
    "create_dataset"
]


def rmtree(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top)


def create_dataset(dataset_name, preprocess=True):
    data_path = os.path.join(os.path.abspath(""), "data", dataset_name)
    dataset_path = os.path.join(os.path.abspath(""), "data", dataset_name + "_dataset")
    if os.path.exists(dataset_path):
        rmtree(dataset_path)

    os.mkdir(dataset_path)
    os.mkdir(os.path.join(dataset_path, "train"))
    os.mkdir(os.path.join(dataset_path, "val"))
    os.mkdir(os.path.join(dataset_path, "test"))

    for subject in os.listdir(data_path):
        os.mkdir(os.path.join(dataset_path, "train", subject))
        os.mkdir(os.path.join(dataset_path, "val", subject))
        os.mkdir(os.path.join(dataset_path, "test", subject))

        train_images, val_images = train_test_split(os.listdir(os.path.join(data_path, subject)), test_size=0.1, random_state=42)
        train_images, test_images = train_test_split(train_images, test_size=0.05, random_state=42)
        for part, images in [("train", train_images), ("val", val_images), ("test", test_images)]:
            for image in images:
                print(os.path.join(data_path, subject, image) + " -- " + os.path.join(dataset_path, part, subject, image))
                if preprocess:
                    img = cv2.imread(os.path.join(data_path, subject, image))
                    cv2.imwrite(os.path.join(dataset_path, part, subject, image), align_image(img))
                else:
                    shutil.copyfile(os.path.join(data_path, subject, image),
                                    os.path.join(dataset_path, part, subject, image))


if __name__ == "__main__":
    create_dataset("vggface2_train")
    create_dataset("vggface2_test")
