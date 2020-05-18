from . import BuildModel_Simple, BuildModel_UNet, NeuralNet
import re
from config import *
from tensorflow.keras import backend as K


class BuildModel(BuildModel_UNet, BuildModel_Simple):
    def __init__(self, name, inp_shape, deepness, filters, path):
        try:
            re_unet = r"unet"
            re_conv = r"conv"
            if re.match(re_unet, name, re.IGNORECASE):
                print('nemjó')
                BuildModel_UNet.__init__(self, inp_shape, deepness, filters)
            elif re.match(re_conv, name, re.IGNORECASE):
                print('nnnna')
                BuildModel_Simple.__init__(self, inp_shape, deepness, filters)
            self.saveModel(path)
        except EOFError as err:
            print('Building is not succeeded because: ', err)


def mse_corr(y_actual, y_predicted):

    loss_func = K.sum(K.square(K.cast((y_predicted - y_actual), 'float32'))) / \
                K.sum(K.cast(K.greater(y_actual, THRESHOLD), 'float32'))
    return loss_func

def modelUse(model, images, train, test):
    name = model["name"] + ".h5"
    path_ = os.path.join(PATH, 'models', name)
    net = NeuralNet()
    net.loadModel(path_)
    net.model.compile(optimizer='adam', loss=mse_corr, metrics=['accuracy'])
    image_to_use = images[model["images"]]
    if train:
        net.trainModel(image_to_use, image_to_use, model['batch_size'], 3)
        net.saveTrained(path_, model)
    if test:
        net.predictModel(image_to_use, image_to_use, model["batch_size"])
        # net.showPrediction(net.predicted, model['test_pics'])
        big_losses = [(index, losses) for index, losses in enumerate(net.loss) if (losses >= model['loss'][-1])]
        big_loss = []
        for i in big_losses:
            big_loss.append({"index": i[0], "loss": i[1]})
        model["test"][str(model["images"])] = {"loss": net.loss, "accuracy": net.accuracy,
                                               "bigger_losses": big_loss}
