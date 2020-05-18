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

        conv1 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
        norm1 = keras.layers.BatchNormalization()(conv1)
        conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(norm1)
        norm2 = keras.layers.BatchNormalization()(conv2)
        conv3 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(norm2)
        drop = keras.layers.Dropout(0.5)(conv3)
        conv4 = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(drop)

        flat = keras.layers.Flatten()(conv4)
        output_layer = keras.layers.Dense(self.classes, activation='softmax')(flat)
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.model.build(input_shape=self.input_shape)

        print(self.model.summary())
        self.saveModel(path)

        # self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    # def train_network(self, input_labels, batch_size, epoch):
    #     start_time = time.time()
    #     history = self.model.fit(self.input_images, input_labels, batch_size=batch_size, epochs=epoch)
    #     self.time = time.time() - start_time
    #     self.loss = history.history['loss']
    #     self.accuracy = list(map(float, history.history['accuracy']))
