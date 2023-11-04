import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from utils.model.create_model import Model
from utils import log


class ModelTesting:
    """ class for testing the model """

    def __init__(self, visualize=False) -> None:

        model_name = "saved_models/cnn_mnist_model.keras"
        if not os.path.isfile(model_name):
            log.warn("Model not detected. Creating model.", "!")
            model = Model(model_name, epochs=5)
            model.create()

        self.model = keras.models.load_model(model_name)
        self.dataset = keras.datasets.mnist
        self.visual = visualize

    def test(self) -> None:
        """ 
        Tests the model

        :return: None
        """

        mnist = self.dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_test = keras.utils.normalize(x_test, axis=1)
        predictions = self.model.predict([x_test])

        correct_val = 0
        inc = 0
        for x in range(len(predictions)):
            inc += 1
            content = f"Program prediction: {np.argmax(predictions[x])}\nAnswer:{y_test[x]} ({inc}/{len(predictions)})"
            if (np.argmax(predictions[x])) == y_test[x]:
                log.success(content)
                correct_val += 1

            else:
                log.error(content)

            if self.visual:
                plt.imshow(x_test[x], cmap=plt.cm.binary)
                plt.show()

        log.info(f"Score: {correct_val}/{len(predictions)} ({int(correct_val / len(predictions) * 100)}% Accuracy)")

    def predict(self, data) -> int:
        """ 
        Predicts the number 

        :param data:
        :return: int
        """

        data = keras.utils.normalize(data, axis=1)
        predictions = self.model.predict(data)
        print(predictions)
        return np.argmax(predictions[0])
