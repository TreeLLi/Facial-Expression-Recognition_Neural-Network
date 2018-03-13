from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform
import glob

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }



def load_FER2013_samples(filepath, amount=None):
    samples = []
    paths = glob.glob(os.path.join(filepath, '*.jpg'))
#    samplepaths = sorted(paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    samplepaths = sorted(paths, key=lambda i: os.path.splitext(os.path.basename(i))[0])
    i = 0
    for samplepath in samplepaths:
        # print(samplepath)
        img = imread(samplepath, True)
        samples.append(img)
        i += 1
        if amount is not None and i==amount:
            break
    return np.array(samples)

def load_FER2013_labels(filepath, train_amount=None, test_amount=None):
    train_labels = []
    test_labels = []
    i = 0
    with open(filepath, "r") as labelsfile:
        lines = labelsfile.readlines()
        train_lines = lines[1:28710]
        test_lines = lines[28710:]
        for line in train_lines:
            train_labels.append(int(line.split(",")[-1]))
            i += 1
            if train_amount is not None and i==train_amount:
                break

        i = 0
        for line in test_lines:
            test_labels.append(int(line.split(",")[-1]))
            i += 1
            if test_amount is not None and i==test_amount:
                break
        return  np.array(train_labels), np.asarray(test_labels)

    
def get_FER2013_data(num_train=None, num_valid=1, num_test=None):
    path = {
        "train_samples" : "datasets/FER2013/Train",
        "test_samples" : "datasets/FER2013/Test",
        "labels" : "datasets/FER2013/labels_public.txt"
    }

    num_train_valid = num_train + num_valid if num_train is not None else None
    train_samples = load_FER2013_samples(path["train_samples"], num_train_valid)
    test_samples = load_FER2013_samples(path["test_samples"], num_test)
    train_labels, test_labels = load_FER2013_labels(path["labels"], num_train_valid, num_test)

    return split_dataset(train_samples,
                         test_samples,
                         train_labels,
                         test_labels,
                         num_valid)

def get_FER2013_data_from_binary(num_valid=1):
    path = {
        "train_samples" : "datasets/fer2013_npy/train_samples.npy",
        "train_labels" : "datasets/fer2013_npy/train_labels.npy",
        "test_samples" : "datasets/fer2013_npy/test_samples.npy",
        "test_labels" : "datasets/fer2013_npy/test_labels.npy"
    }

    train_samples = np.load(open(path["train_samples"], 'rb'))
    train_labels = np.load(open(path["train_labels"], 'rb'))
    test_samples = np.load(open(path["test_samples"], 'rb'))
    test_labels = np.load(open(path["test_labels"], 'rb'))

    return split_dataset(train_samples,
                         test_samples,
                         train_labels,
                         test_labels,
                         num_valid)
    
def split_dataset(train_samples, test_samples, train_labels, test_labels, num_valid):
    real_train_num = len(train_samples) - num_valid
    data = {
        "X_train": train_samples[ : real_train_num],
        "y_train": train_labels[ : real_train_num],
        "X_val": train_samples[real_train_num : ],
        "y_val": train_labels[real_train_num : ],
        "X_test": test_samples,
        "y_test": test_labels,
    }
    
    print ("#####################################################################")
    print ("The loaded dataset:")
    for key, value in data.items():
        print (key, value.shape[0])
    print ("#####################################################################\n")

    return data
