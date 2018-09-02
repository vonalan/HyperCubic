import warnings

import mxnet as mx
from mxnet.gluon.nn import HybridSequential, HybridBlock
from mxnet.gluon.nn import Dense, Conv2D


class YoloHead(HybridBlock):
    def __init__(self, num_boxes_per_grid, num_classes, prefix=None, **kwargs):
        super(YoloHeadv1, self).__init__(prefix=prefix, **kwargs)

        self.dense_4096 = Dense(4096, activation='lrelu', prefix=prefix+'dense_4096/')

        channels = num_boxes_per_grid * (4 + 1) + num_classes
        self.dense_vary = Dense(7 * 7 * channels, prefix=prefix+'dense_vary/')


    def hybrid_forward(self, F, x, *args, **kwargs):
        '''

        :param F:
        :param x: [N, 1024, 7, 7]
        :param args:
        :param kwargs:
        :return:
        '''
        F = mx.symbol

        x = F.reshape(x, shape=(-1, 1024 * 7 * 7))
        x = self.dense_4096(x)
        x = self.dense_vary(x)
        x = F.reshape(x, shape=(-1, 1024, 7, 7))

        return x
