import warnings

import mxnet as mx
from mxnet.gluon.nn import HybridSequential, HybridBlock
from mxnet.gluon.nn import Dense, Conv2D


class YoloHead(HybridBlock):
    def __init__(self, num_anchors, num_classes, prefix=None, **kwargs):
        '''

        :param num_anchors:
        :param num_classes:
        :param prefix:
        :param kwargs:
        '''
        super(YoloHead, self).__init__(prefix=prefix, **kwargs)

        channels = num_anchors * (4 + 1 + num_classes)
        self.conv2d_1x1 = Conv2D(channels, (1,1), strides=(1,1), padding=(0,0), prefix=prefix+'conv2d_1x1/')

    def hybrid_forward(self, F, x, *args, **kwargs):
        '''

        :param F:
        :param x:
        :param args:
        :param kwargs:
        :return: tx, ty, tw, th, to, t_classes
        '''
        x = self.conv2d_1x1(x)

        return x


class YoloOut(HybridBlock):
    def __init__(self, anchors, prefix=None, **kwargs):
        super(YoloOut, self).__init__(prefix=prefix, **kwargs)


class YoloLoss(HybridBlock):
    def __init__(self, anchors, coords_scale=5.0, noobj_scale=0.5, class_scale=1.0, prefix=None, **kwargs):
        '''

        :param anchors: list of anchors, anchor -- (pw, ph)
        :param prefix:
        :param kwargs:
        '''
        super(YoloLoss, self).__init__(prefix=prefix, **kwargs)
        self.coord_scale = coords_scale
        self.noobj_scale = noobj_scale
        self.class_scale = class_scale

        # anchors are the part of network
        self.anchors = self.params.get_constant('anchors', mx.nd.array(anchors))

    def hybrid_forward(self, F, x, *args, **kwargs):
        '''

        :param F:
        :param x: yolo_out -- [N, 5 * (4 + 1 + 20), 13, 13]
        :param args:
        :param kwargs:
        :return: loss
        '''
        F = mx.symbol

        gt_bboxes, positive_iou_threshold = args
        anchors = self.anchors.data()
        num_anchors = anchors.shape[0]

        N, C, H, W = x.shape
        x = F.reshape(x, shape=(N, num_anchors, -1, H, W))
        txy = x[:,:,0:2,:,:]
        twh = x[:,:,2:4,:,:]
        to = x[:,:,4:5,:,:]
        tc = x[:,:,5:,:,:]

        # todo: debug
        cxy, pwh = self.gen_anchor_boxes(anchors, W, H)

        # todo: 
        bxy = F.sigmoid(txy) + cxy
        bwh = F.exp(twh) * pwh
        bxywh = [bxy, bwh]
        anchor_boxes = F.Concat(*bxywh, dim=2)
        anchor_boxes = F.reshape(anchor_boxes, shape=(N, 5, 4, H, W))
        anchor_boxes = F.transpose(anchor_boxes, axis=(0,1,3,4,2))
        anchor_labels = F.softmax(tc, axis=2)
        anchor_masks = F.sigmoid(to)

        gt_boxes = gt_bboxes[:,:4]
        _, gt_labels, gt_masks = \
            self.gen_anchor_targets(gt_bboxes, anchor_boxes, positive_iou_threshold=positive_iou_threshold)

        reg_loss = self.calc_reg_loss(gt_boxes, anchor_boxes, anchor_masks)
        cls_loss = self.calc_cls_loss(gt_labels, anchor_labels, anchor_masks)
        con_loss = self.calc_con_loss(gt_masks, anchor_masks)

        loss = reg_loss + cls_loss + con_loss

        return loss

    def gen_anchor_boxes(self, anchors, W, H):
        '''

        :param H:
        :param W:
        :return:
        '''
        F = mx.symbol

        # TODO: debug, optimize
        x = mx.nd.array([[i for i in range(W)]])
        x = F.repeat(x, repeats=H, axis=0)

        y = mx.nd.array([[i] for i in range(H)])
        y = F.repeat(y, repeats=W, axis=1)

        xy = F.Concat(F.expand_dims(x, axis=0), F.expand_dims(y, axis=0), dim=0)

        wh = anchors
        wh = F.expand_dims(wh, axis=2)
        wh = F.expand_dims(wh, axis=3)
        wh = F.repeat(wh, repeats=H, axis=2)
        wh = F.repeat(wh, repeats=W, axis=3)

        return xy, wh

    def gen_anchor_targets(self, gt_bboxes, anchor_boxes, positive_iou_threshold=0.5):
        '''
        assign label and mask to each anchor box

        :param gt_boxes: [N, max_num_bboxes, 5], [x,y,w,h,label]
        :param num_anchor_boxes: [N, num_anchor_boxes, 4], [x,y,w,h]
        :param positive_iou_threshold:
        :return:
        '''
        warnings.warn('please check the positive_iou_threshold ... ')

        # TODO: debug, convert to symbol
        # TODO: only a anchor box is responsible to predict a ground truth
        N, M = gt_bboxes.shape[0], anchor_boxes.shape[0]
        anchor_labels = []
        anchor_masks = []
        for i in range(gt_bboxes.shape[0]):
            _labels = []
            _masks = []
            for j in range(anchor_boxes.shape[0]):
                if(self.iou(gt_bboxes[i][:4], anchor_boxes[j][:4]) >= positive_iou_threshold):
                    _labels.append(gt_bboxes[i][4])
                    _masks.append(1)
                else:
                    _labels.append(0)
                    _masks.append(0)
            anchor_labels.append(_labels)
            anchor_masks.append(_masks)

        return anchor_boxes, anchor_labels, anchor_masks

    def iou(self, gt_boxes, anchor_boxes):
        '''

        :param gt_boxes: [x,y,w,h]
        :param anchor_boxes: [x,y,w,h]
        :return:
        '''
        F = mx.nd

        b1_xy, b1_wh = gt_boxes[:,2], anchor_boxes[:,2:]
        b1_x1y1 = b1_xy - b1_wh / 2.0
        b1_x2y2 = b1_xy + b1_wh / 2.0

        b2_xy, b2_wh = gt_boxes[:,2], anchor_boxes[:,2:]
        b2_x1y1 = b2_xy - b2_wh / 2.0
        b2_x2y2 = b2_xy + b2_wh / 2.0

        x1y1 = F.maximum(b1_x1y1, b2_x1y1)
        x2y2 = F.minimum(b1_x2y2, b2_x2y2)

        wh = x2y2 - x1y1
        wh = F.maximum(wh, 0)

        i = wh[:,0] * wh[:,1]
        u = b1_wh[:,0] * b1_wh[:,1] + b2_wh[:,0] * b2_wh[:,1] - i

        iou = i / u

        return iou

    def calc_reg_loss(self, gt_boxes, anchor_boxes, anchor_masks):
        '''

        :param gt_boxes:
        :param anchor_boxes:
        :param anchor_masks:
        :return:
        '''
        F = mx.symbol

        box1_xy, box1_wh = gt_boxes[:,2], gt_boxes[:,2:4]
        box2_xy, box2_wh = anchor_boxes[:,2], anchor_boxes[:,2:4]

        loss_xy = F.sum(F.square(box1_xy - box2_xy) * anchor_masks)
        loss_wh = F.sum(F.square(F.sqrt(box1_wh) - F.sqrt(box2_wh)) * anchor_masks)

        loss = loss_xy + loss_wh
        loss = self.coord_scale * loss

        return loss


    def calc_cls_loss(self, gt_labels, anchor_labels, anchor_masks):
        '''

        :param gt_labels:
        :param anchor_labels:
        :param anchor_masks:
        :return:
        '''
        F = mx.symbol

        # TODO: DEBUG
        # todo: still use mean square error
        loss = F.softmax_cross_entropy(anchor_labels, gt_labels) * anchor_masks
        loss = F.mean(loss)
        loss = self.class_scale * loss

        return loss


    def calc_con_loss(self, gt_masks, anchor_masks):
        '''

        :param gt_masks:
        :param anchor_masks:
        :return:
        '''
        F = mx.symbol

        # todo: v1 --> v2
        # todo: still use mean square error
        loss_raw = F.softmax_cross_entropy(anchor_masks, gt_masks)
        loss_obj = F.sum(loss_raw * anchor_masks)
        loss_noobj = F.sum(loss_raw * (1-anchor_masks))
        loss = 1.0 * loss_obj + self.noobj_scale * loss_noobj

        return loss


if __name__ == '__main__':
    net = HybridSequential()
    net.add(YoloHead(5, 4, prefix='yolo_head/'))
    net.initialize()

    x = mx.nd.random_uniform(-1,1,shape=(2,1024,13,13))
    y = net(x)
    print(y.shape, y.mean())
