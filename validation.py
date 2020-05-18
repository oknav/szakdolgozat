import random
import sys, os
from os.path import abspath as abs
from data import Images
import json
import numpy as np
from network.classifier import Classifier
from valid.num_on_pic import Draw
from config import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


def validate(classify):
    label_path = abs(os.path.join(PATH, "labels.json"))

    imgs = Images(NPY_PATH)
    # Draw on image
    if DRAW_IMAGE:
        labels = []
        arrays = []
        for i in range(VALID_IMG_NUM):
            number = random.randint(0, CLASSES-1)
            img_name = "image_%s.png" % i
            img_path = abs(os.path.join(VALID_PATH, img_name))
            labels.append(number)
            draw = Draw(imgs.arr[i], number, img_path)
            arrays.append(draw.array)

        arrays = np.array(arrays)
        try:
            with open(label_path, 'r') as file:
                j_son = json.load(file)
            if HOMOGENE:
                j_son['homogene'] = labels
            elif RANDOM:
                j_son['random'] = labels
            elif KEZEK:
                j_son['kezek'] = labels

            arrays = np.expand_dims(arrays, axis=3)
            np.save(NPY_PATH, arrays)
            try:
                with open(label_path, 'w+') as file:
                    json.dump(j_son, file, indent=4)

            except EOFError:
                print("There is no valid .json file.")
                sys.exit()

        except EOFError:
            print("There is no valid .json file.")
            exit()
    # Classifier
    if classify == 'True':
        images = Images(NPY_PATH)
        # part = int((images.arr.shape[0]) * 3 / 4)
        train_images = np.array([])
        part = 300
        if part != 0:
            train_images = images.arr[:part]
        test_images = images.arr[part:]
        try:
            with open(label_path, 'r') as file:
                j_son = json.load(file)
                label = []
                if HOMOGENE:
                    label = j_son['homogene']
                if RANDOM:
                    label = j_son['random']
                if MIX:
                    label = j_son['mix']
                if KEZEK:
                    label = j_son['kezek']
        except:
            print('Something went wrong.')

        train_labels = list()
        if part != 0:
            train_labels = to_categorical(label[:part])
        test_labels = to_categorical(label[part:])
        classifier = Classifier(imgs.arr.shape[1:], CLASSES)
        network_name = 'classifier.h5'
        network_path = os.path.join(PATH, network_name)
        if not find(network_name, PATH):
            classifier.build_classifier(network_path)
        if HOMOGENE:
            trained_name = 'hom_class.h5'
            trained_path = os.path.join(PATH, trained_name)
        elif RANDOM:
            trained_name = 'rand_class.h5'
            trained_path = os.path.join(PATH, trained_name)
        else:
            trained_path = None

        if find(trained_name, PATH):
            classifier.loadModel(trained_path)
            classifier.model.compile(optimizer=Adam(lr=0.001), loss=LOSS, metrics=['categorical_accuracy'])
        elif train_images.size != 0:
            classifier.loadModel(network_path)
            classifier.model.compile(optimizer=Adam(lr=0.001), loss=LOSS, metrics=['categorical_accuracy'])
            classifier.trainModel(train_images, train_labels, 2, EPOCH)


        json_model = {}

        try:
            with open(os.path.join(PATH, 'classific.json'), 'r') as file:
                j_son = json.load(file)
                if HOMOGENE:
                    json_model = j_son['homogene']
                elif RANDOM:
                    json_model = j_son['random']


        except:
            print('Something went wrong.')

        if train_images.size != 0:
            classifier.saveTrained(trained_path, json_model['train'])
        classifier.predictModel(test_images, test_labels, 1)
        print("itt a baj")
        test = []
        print(classifier.predicted)
        for i, label in enumerate(test_labels):
            test_ = dict()
            test_[str(np.argmax(label))] = str(np.argmax(classifier.predicted[i]))
            test.append(test_)
        json_model['test'] = test
        try:
            with open(os.path.join(PATH, 'classific.json'), 'w') as file:
                if HOMOGENE:
                    j_son['homogene'] = json_model
                elif RANDOM:
                    j_son['random'] = json_model
                elif MIX:
                    j_son['mix'] = json_model
                elif KEZEK:
                    j_son['kezek'] = json_model

                json.dump(j_son, file, indent=4)

        except EOFError:
                print("Save to .json file was unsuccessful.")
    else:
        pass

