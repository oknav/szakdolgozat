import numpy as np


class Images:
    def __init__(self, path):
        self.arr = None
        self.inp_shape = None
        self.setData(path)

    def setData(self, path):
        self.arr = np.load(path)
        self.arr[self.arr <= 7] = 0.0
        self.arr = self.arr.astype('float32')
        self.inp_shape = np.shape(self.arr)[1:]

    def partData(self, pics_in_one_batch):
        partitions = {}
        sizes = []
        for size in range(0, len(self.arr)+1, pics_in_one_batch):
            sizes.append(size)
        remaining = len(self.arr) % pics_in_one_batch
        if remaining:
            last_size = sizes[-1]
            sizes.append(last_size + remaining)
        for i in range(1, len(sizes)):
            part = self.arr[sizes[i-1]:sizes[i]]
            partitions[i] = part
        return partitions
