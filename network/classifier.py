from tensorflow import keras
import time
from .network import NeuralNet


class Classifier(NeuralNet):
    def __init__(self, input_shape, classes):
        NeuralNet.__init__(self)
        self.input_shape = input_shape
        self.classes = classes

    def build_classifier(self, path):
        input_layer = keras.layers.Input(shape=self.input_shape)

        conv1 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_layer)
        norm1 = keras.layers.BatchNormalization()(conv1)
        rel1 = keras.layers.Activation('relu')(norm1)
        pool1 = keras.layers.MaxPool2D()(rel1)
        conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(pool1)
        norm2 = keras.layers.BatchNormalization()(conv2)
        rel2 = keras.layers.Activation('relu')(norm2)
        pool2 = keras.layers.MaxPool2D()(rel2)
        conv3 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(pool2)
        norm3 = keras.layers.BatchNormalization()(conv3)
        rel3 = keras.layers.Activation('relu')(norm3)
        pool3 = keras.layers.MaxPool2D()(rel3)
        conv4 = keras.layers.Conv2D(filters=288, kernel_size=(3, 3), padding='same')(pool3)
        norm4 = keras.layers.BatchNormalization()(conv4)
        rel4 = keras.layers.Activation('relu')(norm4)
        pool4 = keras.layers.MaxPool2D()(rel4)
        flat = keras.layers.Flatten()(pool4)
        output_layer = keras.layers.Dense(self.classes, activation='softmax')(flat)
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.model.build(input_shape=self.input_shape)

        print(self.model.summary())
        self.saveModel(path)

