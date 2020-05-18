from PIL import Image
from data.image import Images
import numpy as np
import os
from config import *


class Pictures(Images):
    def __init__(self, img_path):
        Images.__init__(self, img_path)
        self.image_path = img_path

    def save_pictures(self, number):
        image = self.arr[number]
        image = np.reshape(image, image.shape[:2])
        img = Image.fromarray(image)
        name = 'image_number_%s.png' % number
        path = os.path.join(TRAIN_FOLDER, name)
        img = img.convert('RGB')
        img.save(path, format='png')

    def cut_images(self, images):
        if images:
            self.arr = np.delete(self.arr, images, axis=0)
            np.random.shuffle(self.arr)
            np.save(self.image_path, self.arr)
        else:
            pass
