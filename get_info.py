from data.save_results import DataHandler
from data.pictures import Pictures
from config import *
import os


def analyze(train_number):
    js = DataHandler(JSON_PATH, NPY_PATH)
    pics = Pictures(NPY_PATH)

    cut = CUT

    js.save_csv(train_number, cut)
    cut_file = os.path.join(TRAIN_FOLDER, 'images_cut.txt')
    with open(cut_file) as f:
        images_to_save = f.read().splitlines()
        images_to_save = list(map(int, images_to_save))
        for i in images_to_save:
            pics.save_pictures(i)
        pics.cut_images(images_to_save)
