import tensorflow as tf 
from tensorflow import keras 

from utils import log

class Model:
    """ class for creating & training a model and saving it """

    def __init__(
        self,
        model_name: str,
        neurons:    int = 128,
        epochs:     int = 3
    ) -> None:

        self.model_name = model_name
        self.dataset = keras.datasets.mnist 
        self.neurons = neurons
        self.epochs = epochs
   
    def gpu_check(self):
        """ checks if the program will run on the GPU """

        tfGraphic = tf.config.list_physical_devices("GPU")
        if not tfGraphic:
            log.error("The program will be running on the cpu")
            return True
        else:
            log.info("Program is running on GPU(s)")
            return False

    def create(self):
        self.gpu_check()

        mnist = self.dataset
        (x_train,y_train), (x_test,y_test) = mnist.load_data()

        # Rescaling the info to 0-1
        x_test = keras.utils.normalize(x_test,axis=1)
        x_train = keras.utils.normalize(x_train,axis=1)

        seq_model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),   
            keras.layers.Dense(self.neurons,activation=tf.nn.relu),  #  Hidden Layer
            keras.layers.Dense(self.neurons, activation=tf.nn.relu), #  Hidden Layer
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.model = seq_model

        seq_model.compile(
            optimizer="adam",
            metrics=["accuracy"],
            loss=["sparse_categorical_crossentropy"]
        )

        seq_model.fit(x=x_train,y=y_train,epochs=self.epochs)

        seq_model.save(self.model_name)




