import numpy as np
import os
from keras.models import load_model
from pickle import load

from src.utils.data_utils import load_FER2013_samples

from src.fcnet import FullyConnectedNet

train_samples_mean_path = 'datasets/fer2013_npy/train_samples_mean.npy'

def test_fer_model(img_folder, model="/path/to/model"):
    """
    Given a folder with images, load the images and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """
    preds = None
    ### Start your code here

    if not os.path.exists(model):
        print ("Model Loading Error: can't find the model.\n")
        return None

    if not os.path.exists(img_folder):
        print ("Data Loading Error: can't find the data.\n")
        return None

    with open(model, 'rb') as model_file:
        model = load(model_file)
        data = load_FER2013_samples(img_folder)
        preds = model.predict(data)
        print (preds)
    ### End of code
    return preds


def test_deep_fer_model(img_folder, model_path="/path/to/model"):
    """
    Given a folder with images, load the images and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """
    preds = None
    ### Start your code here

    if not os.path.exists(model_path):
        print ("Model Loading Error: can't find the model.\n")
        return None

    if not os.path.exists(img_folder):
        print ("Data Loading Error: can't find the data.\n")
        return None

    train_samples_mean = np.load(open(train_samples_mean_path, 'rb'))

    test_samples = load_FER2013_samples(img_folder)
    test_samples = test_samples.reshape(-1, 48, 48, 1)
    print(test_samples.shape)
    test_samples = test_samples.astype('float32')
    test_samples /= 255
    test_samples -= train_samples_mean

    model = load_model(model_path)
    model.summary()

    preds = model.predict(test_samples, verbose=1)
    preds = np.argmax(preds, axis=1)
    ### End of code
    return preds
