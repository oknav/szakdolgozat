import os
from os.path import abspath as abs


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files or name in dirs:
            return True
        else:
            return False


PATH = os.path.dirname(__file__)
JSON_PATH = os.path.join(PATH, 'models.json')
MODELS_PATH = os.path.join(PATH, "models")

HOMOGENE = False
RANDOM = True
MIX = False
KEZEK = False

if HOMOGENE and not MIX and not KEZEK:
    VALID_PATH = abs(os.path.join(PATH, 'random'))
elif RANDOM and not MIX and not KEZEK:
    VALID_PATH = os.path.join(PATH, 'homogene')
elif MIX:
    NPY_PATH = os.path.join(PATH, 'mix_test.npy')
elif KEZEK:
    VALID_PATH = abs(os.path.join(PATH, 'fashion'))
    NPY_PATH = abs(os.path.join(VALID_PATH, 'kezek.npy'))

else:
    VALID_PATH = os.path.join(PATH, 'images')

if not MIX and not KEZEK:
    NPY_PATH = os.path.join(VALID_PATH, 'bone_spects.npy')

TRAIN = ["all", "all"]
TEST = ["all", "all"]

LOSS = "categorical_crossentropy"
EPOCH = 5
THRESHOLD = 2


class Config:
    TRAIN_NUMBER = 1
    TRAIN_NAME = 'train_%s' % TRAIN_NUMBER
    TRAIN_FOLDER = os.path.join(os.pardir, 'train_%s' % TRAIN_NUMBER)

NUM_OF_TRAININGS = 1

CUT = 20

DRAW_IMAGE = False
VALID_IMG_NUM = 200
CLASSES = 3
