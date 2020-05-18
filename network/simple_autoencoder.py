import tensorflow as tf
from .network import NeuralNet

class BuildModel_Simple(NeuralNet):
    """
    Build a customized deep autoencoder with the given filter numbers.
    Arguments:
        inp_shape: a tuple, the shape of the input images
        deepness: an integer, the desired deepness.
        filters: a list of the number of filters in the layer's order.
    """
    def __init__(self, inp_shape, deepness, filters):
        NeuralNet.__init__(self)
        self.build_simple(inp_shape, deepness, filters)

    def build_simple(self, inp_shape, deepness, filters):
        tf.keras.backend.clear_session()
        self.model = None

        # Check if the deepness is valid. If not, the deepness is set to the deepest.
        deepness = self.validDeepness(inp_shape, deepness)

        # Create list of the layers name (down,up).
        down_layers = []
        up_layers = []
        valids = []

        inp_layer = tf.keras.layers.Input(shape=inp_shape)
        down_layers.append(inp_layer)

        for i in range(deepness):
            layer = "conv_down_" + str(i)
            down_layers.append(layer)
            layer = "pool_" + str(i)
            down_layers.append(layer)

            layer = "conv_up_" + str(i)
            up_layers.append(layer)
            layer = "up_" + str(i)
            up_layers.append(layer)

        # Build the encoder part and check if the maxpool's result is an odd number or not. If it is so, the padding is
        # set to valid instead of same and then it saves the
        # layer's number into the list called valids.
        pixels = inp_shape[0]
        # ENCODER
        for j, i in enumerate(range(1, len(down_layers), 2)):
            if ((pixels / 2) % 2) and pixels != 2:
                down_layers[i] = tf.keras.layers.Conv2D(filters=filters[j], kernel_size=(3, 3), padding='valid',
                                                        activation='relu')(down_layers[i - 1])
                valids.append(i)
            else:
                down_layers[i] = tf.keras.layers.Conv2D(filters=filters[j], kernel_size=(3, 3), padding='same',
                                                        activation='relu')(down_layers[i - 1])

            down_layers[i] = tf.keras.layers.Conv2D(filters=filters[j], kernel_size=(3, 3), padding='same',
                                                    activation='relu')(down_layers[i])
            down_layers[i + 1] = tf.keras.layers.MaxPool2D()(down_layers[i])
            pixels = int(down_layers[i + 1].get_shape()[1])
            # print(down_layers[i], down_layers[i+1])

        # BOTTLE_NECK
        drop = tf.keras.layers.Dropout(0.3)(down_layers[-1])
        up_layers.append(drop)

        # Build the decoder part of the network. If the layer number is in valids, it uses valid padding,
        # otherwise it uses same. Also it concatenate the right layers.

        # DECODER
        for j, i in enumerate(range(len(up_layers) - 2, -1, -2)):
            up_layers[i] = tf.keras.layers.UpSampling2D()(up_layers[i + 1])
            if i in valids:
                up_layers[i - 1] = tf.keras.layers.Conv2D(filters=filters[deepness - 1 - j],
                                                                   kernel_size=(3, 3), padding='valid',
                                                                   activation='relu')(up_layers[i])
            else:
                up_layers[i - 1] = tf.keras.layers.Conv2D(filters=filters[deepness - 1 - j], kernel_size=(3, 3),
                                                          padding='same', activation='relu')(up_layers[i])

            up_layers[i - 1] = tf.keras.layers.Conv2D(filters=filters[deepness - 1 - j], kernel_size=(3, 3),
                                                      padding='same', activation='relu')(up_layers[i - 1])

        out_layer = tf.keras.layers.Conv2D(filters=inp_shape[2], kernel_size=(3, 3), padding='same', activation='relu')(
            up_layers[0])

        self.model = tf.keras.models.Model(inputs=inp_layer, outputs=out_layer)
        self.model.build(input_shape=inp_shape)
        print(self.model.summary())
