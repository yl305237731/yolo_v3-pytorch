import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np


def anchor_iou(box, anchor):
    """
    Move the box and anchor box to the top left corner to overlap, calculate IOU
    :param box:
    :param anchor:
    :return:
    """

    def iou(box_a, box_b):
        area_boxa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_boxb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        def intersection(box1, box2):
            x_lt = max(box1[0], box2[0])
            y_lt = max(box1[1], box2[1])
            x_br = min(box1[2], box2[2])
            y_br = min(box1[3], box2[3])
            inter_w = max(x_br - x_lt, 0)
            inter_h = max(y_br - y_lt, 0)
            return float(inter_w * inter_h)

        area_inter = intersection(box_a, box_b)
        return area_inter / (area_boxa + area_boxb - area_inter + 1e-5)

    box = [0, 0, box[2] - box[0], box[3] - box[1]]
    anchor = [0, 0, anchor[0], anchor[1]]
    return iou(box, anchor)


def adjust_learning_rate(initial_lr, optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = 3
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class CosineDecayLR(object):
    def __init__(self, optimizer, T_max, lr_init, lr_min=0., warmup=0):
        """
        a cosine decay scheduler about steps, not epochs.
        :param optimizer: ex. optim.SGD
        :param T_max:  max steps, and steps=epochs * batches
        :param lr_max: lr_max is init lr.
        :param warmup: in the training begin, the lr is smoothly increase from 0 to lr_init, which means "warmup",
                        this means warmup steps, if 0 that means don't use lr warmup.
        """
        super(CosineDecayLR, self).__init__()
        self.__optimizer = optimizer
        self.__T_max = T_max
        self.__lr_min = lr_min
        self.__lr_max = lr_init
        self.__warmup = warmup


    def step(self, t):
        if self.__warmup and t < self.__warmup:
            lr = self.__lr_max / self.__warmup * t
        else:
            T_max = self.__T_max - self.__warmup
            t = t - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (1 + np.cos(t/T_max * np.pi))
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr


def custom_collate_fn(batch):
    items = list(zip(*batch))
    imgs = []
    for _, sample in enumerate(items[0]):
        imgs.append(sample)

    annos = list(items[1])
    return torch.stack(imgs, 0), annos


def to_cpu(x):
    return x.detach().cpu()


def parse_voc_annotation(ann_dir, img_dir, labels=[]):
    all_insts = []
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}
        try:
            tree = ET.parse(os.path.join(ann_dir, ann))
        except Exception as e:
            print('Ignore this bad annotation: ' + ann_dir + ann)
            continue

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_insts += [img]
    return all_insts


def parse_wider_annotation(label_txt_path, img_root):
    all_insts = []
    f = open(label_txt_path, 'r')
    lines = f.readlines()
    is_First = True
    img = {'object': []}
    obj = {}
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            if not is_First:
                if len(img['object']) > 0:
                    all_insts += [img.copy()]
                    img = {'object': []}
                    obj.clear()
            img_path = line[2:]
            if not os.path.exists(os.path.join(img_root, img_path)):
                continue
            image = cv2.imread(os.path.join(img_root, img_path))
            h, w, c = image.shape
            img["filename"] = img_path
            img["width"] = w
            img["height"] = h
            if is_First:
                is_First = False
        else:
            line = line.split(' ')
            x_min, y_min, w, h = float(line[0]), float(line[1]), float(line[2]), float(line[3])

            # [x1, y1, x2, y2, class_index]
            # for widerface, set face as 1, background as 0
            obj["xmin"] = x_min
            obj["ymin"] = y_min
            obj["xmax"] = x_min + w
            obj["ymax"] = y_min + h
            img['object'] += [obj]
    # process last image
    if len(img['object']) > 0:
        all_insts += [img]
    return all_insts
