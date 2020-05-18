import random
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import sys
import numpy as np


class NeuralNet:
    def __init__(self):
        self.model = None
        self.predicted = None
        self.loss = []
        self.accuracy = []
        self.time = None

    def trainModel(self, x, y, batch_size, epoch):
        start_time = time.time()
        history = self.model.fit(x, y, batch_size=batch_size, epochs=epoch)
        self.time = time.time() - start_time
        self.loss = list(map(float, history.history['loss']))
        try:
            self.accuracy = list(map(float, history.history['accuracy']))
        except:
            self.accuracy = list(map(float, history.history['categorical_accuracy']))

    def predictModel(self, test_im, test_lab, batch_size):
        self.predicted = self.model.predict(test_im, batch_size=batch_size)
        if test_im.all == test_lab.all:
            for i, picture in enumerate(test_im):
                picture = np.expand_dims(picture, axis=0)
                scores = self.model.evaluate(picture, picture, verbose=0)
                self.loss.append(float(scores[0]))
                self.accuracy.append(float(scores[1]))
        else:
            # pass
            scores = self.model.evaluate(test_im, test_lab, verbose=0)
            self.loss.append(scores[0])
            self.accuracy.append(float(scores[1]))


    def showPrediction(self, data, number):
        h = int(number/2)
        w = number-h
        if data.shape[-1] == 1:
            size = (data.shape[1], data.shape[2])
        else:
            size = data.shape[1:]

        plt.ion()
        for i in range(1, h*w, 2):
            index = random.randint(0, self.predicted.shape[0])
            plt.figure(figsize=(5, 5))
            plt.subplot(1, 2, 1)
            ImShow(data[index].reshape(size))
            plt.subplot(1, 2, 2)
            ImShow(self.predicted[index].reshape(size))

    def validDeepness(self, shape, deep):
        x = shape[0]
        y = shape[1]
        deepest = 0
        while x > 0 and y > 0:
            deepest += 1
            x = int(x / 2)
            y = int(y / 2)

        if deep > deepest or deep is 0:
            print("WARNING!!! The deepest UNet for the actual input shape is {}.".format(str(deepest)))
            return deepest
        else:
            return deep

    def saveModel(self, path):
        try:
            self.model.save(path)

        except EOFError:
            print('Could not save model.')

    def saveTrained(self, path, json_model):
        try:
            self.model.save(path)
            json_model['train_time'] = self.time
            json_model['accuracy'] = self.accuracy
            json_model['loss'] = self.loss

        except EOFError:
            print('Could not save model.')

    def loadModel(self, path):
        try:
            self.model = tf.keras.models.load_model(path, compile=False)
        except EOFError:
            print("Could not load model.")
            sys.exit()


def ImShow(x):
    plt.imshow(x.astype('uint8'))
