import torch
import cv2
import os
import math
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from tools.utils import anchor_iou


labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
          'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

anchors = [[22, 42], [47, 138], [67, 66], [85, 227], [140, 127], [146, 288], [231, 335], [294, 183], [366, 355]]


class VOCDataset(Dataset):

    def __init__(self, img_root, xml_root, target_size, anchors, name_list, reduction=32, max_box_per_image=30,
                 augmentation=None, transform=None):
        self.anchors = anchors
        self.img_root = img_root
        self.xml_root = xml_root
        self.target_size = target_size
        self.reduction = reduction
        self.name_list = name_list
        self.class_num = len(self.name_list)
        self.transform = transform
        self.max_box_per_image = max_box_per_image
        self.augmentation = augmentation
        self.img_names, self.img_bboxs = self.parse_xml()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_bbox = self.img_bboxs[idx]
        img_ori = cv2.imread(os.path.join(self.img_root, img_name))
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        if self.augmentation:
            img, img_bbox = self.augmentation.augment(img, img_bbox, noise=False, withName=True)
        height, width, _ = img.shape
        img = cv2.resize(img, self.target_size)
        img_bbox = torch.floor(torch.Tensor(img_bbox) * torch.Tensor([self.target_size[0] / width, self.target_size[1] / height,
                                                                      self.target_size[0] / width, self.target_size[1] / height, 1]))
        if self.transform:
            img = self.transform(img)

        yolo1, yolo2, yolo3 = self.encode_target(img_bbox)
        return img, yolo1, yolo2, yolo3

    def encode_target(self, bboxs):
        base_grid_h, base_grid_w = self.target_size[0] / 32, self.target_size[1] / 32
        grid_w = [4 * base_grid_w, 2 * base_grid_w, base_grid_w]
        grid_h = [4 * base_grid_h, 2 * base_grid_h, base_grid_h]

        yolo_1 = torch.zeros(3, 4 + 1 + self.class_num, int(grid_h[2] * grid_w[2]))
        yolo_1_mask = torch.zeros(3, int(grid_h[2] * grid_w[2]))

        yolo_2 = torch.zeros(3, 4 + 1 + self.class_num, int(grid_h[1] * grid_w[1]))
        yolo_2_mask = torch.zeros(3, int(grid_h[1] * grid_w[1]))

        yolo_3 = torch.zeros(3, 4 + 1 + self.class_num, int(grid_h[0] * grid_w[0]))
        yolo_3_mask = torch.zeros(3, int(grid_h[0] * grid_w[0]))

        true_bboxs = [torch.zeros((self.max_box_per_image, 4)) for _ in range(3)]
        bbox_count = torch.zeros((3,))

        yolos = [yolo_3, yolo_2, yolo_1]
        yolo_masks = [yolo_3_mask, yolo_2_mask, yolo_1_mask]

        for box in bboxs:
            max_anchor = None
            max_index = -1
            max_iou = -1

            for i in range(len(self.anchors)):
                anchor = self.anchors[i]
                iou = anchor_iou(box, anchor)
                if max_iou < iou:
                    max_anchor = anchor
                    max_index = i
                    max_iou = iou
            # Small anchors are assigned to the lower levels, and large anchors are assigned to the higher levels
            if max_iou <= 0.2:
                continue
            level = max_index // 3
            yolo = yolos[level]
            mask = yolo_masks[level]
            bbox_ind = int(bbox_count[level] % self.max_box_per_image)
            true_bboxs[level][bbox_ind, :4] = box[:4]
            bbox_count[level] += 1
            anchor_w, anchor_h = max_anchor[0], max_anchor[1]
            w = box[2] - box[0]
            h = box[3] - box[1]
            xc = box[0] + w / 2
            yc = box[1] + h / 2
            w_reduction, h_reduction = self.target_size[0] / grid_w[level], self.target_size[1] / grid_h[level]
            col = math.floor(xc / w_reduction)
            row = math.floor(yc / h_reduction)
            x_offset = xc / w_reduction - col
            y_offset = yc / h_reduction - row
            w_log = torch.log(w / anchor_w)
            h_log = torch.log(h / anchor_h)
            obj_conf = torch.Tensor([1])
            # cls = self.to_onehot(box[4])
            cls = torch.Tensor([1])
            grid_info = torch.cat([x_offset.view(-1, 1), y_offset.view(-1, 1), w_log.view(-1, 1), h_log.view(-1, 1),
                                   obj_conf.view(-1, 1), cls.view(1, -1)], dim=1)
            yolo[max_index % 3, :, int(row * grid_w[level] + col)] = grid_info.clone()
            mask[max_index % 3, int(row * grid_w[level] + col)] = 1
        yolo_1_all = [yolo_1, yolo_1_mask, true_bboxs[2]]
        yolo_2_all = [yolo_2, yolo_2_mask, true_bboxs[1]]
        yolo_3_all = [yolo_3, yolo_3_mask, true_bboxs[0]]
        return yolo_1_all, yolo_2_all, yolo_3_all

    def get_label_index(self, name):
        return self.name_list.index(name)

    def to_onehot(self, clas_idx):
        one_hot = torch.zeros(1, self.class_num)
        one_hot[0, clas_idx.int()] = 1
        return one_hot

    def parse_xml(self):
        img_names = []
        img_bboxs = []
        xml_dir = os.listdir(self.xml_root)

        for xml_name in xml_dir:
            print(xml_name)
            xml_path = os.path.join(self.xml_root, xml_name)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_name = tree.find('filename').text
            if not os.path.exists(os.path.join(self.img_root, img_name)):
                continue
            img_names.append(img_name)
            objs = root.findall('object')
            box_info = list()
            for ix, obj in enumerate(objs):
                name = obj.find('name').text
                if name in self.name_list:
                    box = obj.find('bndbox')
                    x_min = int(float(box.find('xmin').text))
                    y_min = int(float(box.find('ymin').text))
                    x_max = int(float(box.find('xmax').text))
                    y_max = int(float(box.find('ymax').text))
                    label_index = self.get_label_index(name)
                    box_info.append([x_min, y_min, x_max, y_max, label_index])
            if len(box_info) <= 0:
                img_names.remove(img_name)
            else:
                img_bboxs.append(box_info)
        print(len(img_names))
        return img_names, img_bboxs
