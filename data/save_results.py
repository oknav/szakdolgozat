from . import Models, Images
import os
import sys
import pandas
import numpy as np
from config import Config

class DataHandler(Models, Images):
    def __init__(self, path, img_path):
        Models.__init__(self, path)
        Images.__init__(self, img_path)
        self.open()
        self.image_path = img_path

    def get_loss_per_model(self):
        for model in self.models:
            print(model['name'], 'trained on image group', model['images'], ', loss:', model['loss'][-1],
                  ', learning time:', model['train_time'])

    def save_csv(self, number, cut_number):
        if cut_number <= len(self.arr):
            image_losses = dict()
            image_losses['image'] = []
            for i, model in enumerate(self.models):
                all_loss = []
                for image_group in model['test']:
                    loss = [round(num, 3) for num in model['test'][str(image_group)]['loss']]
                    all_loss += loss
                all_loss = [((element-np.mean(all_loss))/np.std(all_loss)) for element in all_loss]
                image_losses[model['name']] = all_loss

            indexes = [index for index in range(len(list(image_losses.values())[1]))]
            image_losses['image'] = indexes
            data_frame = pandas.DataFrame(image_losses, columns=image_losses.keys())
            # all_loss = [((element-np.mean(all_loss))/np.std(all_loss)) for element in all_loss]
            data_frame['sum'] = [sum(row[1:]) for row in data_frame.values.tolist()]
            data_frame.set_index(['image'])
            data_frame = data_frame.sort_values(by=['sum'], ascending=False)

            if not find(TRAIN_NAME, os.pardir):
                os.mkdir(TRAIN_FOLDER)
            print(TRAIN_FOLDER)

            csv_path = os.path.join(TRAIN_FOLDER, 'results_%s.csv' % number)
            data_frame.to_csv(csv_path, index=None, header=True, sep=';', decimal=',')
            cuts = open(os.path.join(TRAIN_FOLDER, 'images_cut.txt'), 'w+')
            cuts.write('\n'.join([str(img) for img in (data_frame['image'][:cut_number])]))
            cuts.close()
        else:
            print('The number is bigger than the number of images.')
            sys.exit()
