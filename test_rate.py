import json
import os
from config import *
from data.image import Images
import numpy as np
from PIL import Image
# from tensorflow.keras import datasets


def get_test_rate(json_path):
    with open(json_path, 'r') as file:
        j_son = json.load(file)

    hom_test = j_son['homogene']['test']
    rand_test = j_son['random']['test']

    success = 0
    for hom in hom_test:
        for key in hom:
            if int(key) == int(hom[key]):
                success += 1

    print("Homogene classification success rate: ", success, "from %s images" % len(hom_test), '(%s)' % (success/len(hom_test)*100))

    success = 0
    for rand in rand_test:
        for key in rand:
            if int(key) == int(rand[key]):
                success += 1

    print("Random classification success rate: ", success, "from %s images" % len(rand_test), '(%s)' % (success/len(rand_test)*100))


def main():
    json_path = os.path.join(PATH, 'classific.json')
    get_test_rate(json_path)


def get_image_shape():
    mix = 'mix_test.npy'
    mix = Images(mix)
    print(mix.arr.shape)


def merge_npy():
    hom = os.path.join(PATH, 'homogene', 'bone_spects.npy')
    rnd = os.path.join(PATH, 'random', 'bone_spects.npy')
    hom = Images(hom)
    rnd = Images(rnd)

    print(hom.arr[300:].shape)
    print(rnd.arr[300:].shape)
    mix = (list(hom.arr[300:]) + list(rnd.arr[300:]))
    mix = np.array(mix, dtype='float32')
    np.save('mix_test.npy', mix)
    print(mix.shape)

    with open('labels.json', 'r') as file:
        j_son = json.load(file)

    labels = j_son['homogene'][300:] + j_son['random'][300:]
    print(len(labels))

    j_son['mix'] = labels
    with open('labels.json', 'w+') as file:
        json.dump(j_son, file, indent=4)


def resize_mnist():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_test = x_test[:200]
    cifar_10 = [Image.fromarray(f_test) for f_test in x_test]
    size = 256, 1024
    resized = [fashion.resize(size) for fashion in cifar_10]
    resized = np.array([np.array(re) for re in resized])
    resized = np.expand_dims(resized, axis=3)
    np.save('fashion/cifar_10.npy', resized)


def show_im():
    img = Images(os.path.join(PATH, 'fashion/cifar_10.npy'))



if __name__ == '__main__':
    main()
