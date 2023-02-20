import os
import pickle
import numpy as np

from PIL import Image, ImageOps
from sklearn.preprocessing import OneHotEncoder

from models import *


def get_model(args):
    """
    Returns neural network model as specified in arguments.
    """
    name = args.model
    if name.lower() == "cnn":
        model = ConvNet()
    else:
        raise Exception(f"Model name {name} is unknown!")
    return model


def load_data_and_labels(datadir="."):
    """
    Loads images from the brain tumor dataset and returns them in array format,
    together with a one-hot-encoded array of their respective class labels.
    """
    labels = []
    data = []

    class_labels = {
        "no_tumor": 0,
        "meningioma_tumor": 1,
        "glioma_tumor": 2,
        "pituitary_tumor": 3,
    }

    # loop over all files, and determine class of image from folder name
    for root, dirs, files in os.walk(datadir):
        for name in files:

            folder_name = os.path.basename(root)
            if ".jpg" in name and folder_name in class_labels.keys():

                filepath = os.path.join(root, name)
                labels.append(class_labels[folder_name])

                # load, resize and normalize image
                im = ImageOps.grayscale(Image.open(filepath))
                im = im.resize((128, 128))
                im_array = np.array(im) / 255.0
                # add one more dimension for the channel
                data.append(im_array[np.newaxis, :])

    assert len(data) == len(labels)

    data = np.array(data).astype(np.float32)
    labels = np.array(labels).astype(np.float32)

    onehot_encoder = OneHotEncoder(sparse=False)
    labels = labels.reshape(len(labels), 1)
    onehot_encoded = onehot_encoder.fit_transform(labels)

    return data, onehot_encoded


def save_pickle(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def datasets_exist():
    return os.path.exists("dataset_test.pkl") and os.path.exists("dataset_train.pkl")
