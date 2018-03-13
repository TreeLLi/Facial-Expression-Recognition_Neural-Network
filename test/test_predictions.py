import numpy as np
import unittest
from sklearn.metrics import confusion_matrix
from src.test import test_fer_model
from src.test import test_deep_fer_model
from src.evaluator import evaluate

test_labels_path = 'datasets/fer2013_npy/test_labels.npy'

class TestFCNNPrediction(unittest.TestCase):

    def test(self):
        data_path = "datasets/FER2013/Test"
        model_path = "models/fcnn.model"
        preds = test_fer_model(data_path, model_path)
        self.assertTrue(preds is not None)

class TestCNNPrediction(unittest.TestCase):

    def test(self):
        data_path = "datasets/FER2013/Test"
        model_path = "models/fer2013_ResNet20v1_model.89-0.66.h5"

        y_pred = test_deep_fer_model(data_path, model_path)
        y_test = np.load(open(test_labels_path, 'rb'))
        print (y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print('Confusion matrix:', cm)

        evaluate(y_pred, y_test, 7, 'resnet_rates')


if __name__ == "__main__":
    unittest.main()
