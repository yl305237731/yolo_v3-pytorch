import torch
import cv2
import os
import math
from torch.utils.data import Dataset
from tools.utils import anchor_iou


labels = ["face"]
anchors = [[2, 4], [4, 8], [7, 13], [12, 24], [20, 35], [35, 56], [56, 96], [103, 151], [180, 237]]


class WiderFaceDataset(Dataset):

    def __init__(self, img_root, label_path, target_size, anchors, reduction=32, max_box_per_image=100, name_list=None,
                 augmentation=None, transform=None):
        if name_list is None:
            name_list = ["face"]
        self.anchors = anchors
        self.img_root = img_root
        self.label_path = label_path
        self.target_size = target_size
        self.reduction = reduction
        self.name_list = name_list
        self.class_num = len(self.name_list)
        self.transform = transform
        self.max_box_per_image = max_box_per_image
        self.augmentation = augmentation
        self.img_names, self.img_bboxs = self.parse_label()

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

        #yolo1: 13 * 13
        yolo_1 = torch.zeros(3, 4 + 1 + self.class_num, int(grid_h[2] * grid_w[2]))
        # mask[:, 0, :] anchor mask, mask[:, 1, :] object mask
        yolo_1_mask = torch.zeros(3, int(grid_h[2] * grid_w[2]))
        #yolo2: 26 * 26
        yolo_2 = torch.zeros(3, 4 + 1 + self.class_num, int(grid_h[1] * grid_w[1]))
        yolo_2_mask = torch.zeros(3, int(grid_h[1] * grid_w[1]))
        #yolo3: 52 * 52
        yolo_3 = torch.zeros(3, 4 + 1 + self.class_num, int(grid_h[0] * grid_w[0]))
        yolo_3_mask = torch.zeros(3, int(grid_h[0] * grid_w[0]))

        # Used to save the real object coordinates corresponding to each yolo head
        # true_bboxs[0]: 52 * 52, ....
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
            # ignore hard gt
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
            # face: 1
            cls = torch.Tensor([1])
            grid_info = torch.cat([x_offset.view(-1, 1), y_offset.view(-1, 1), w_log.view(-1, 1), h_log.view(-1, 1),
                                   obj_conf.view(-1, 1), cls.view(1, -1)], dim=1)
            yolo[max_index % 3, :, int(row * grid_w[level] + col)] = grid_info.clone()
            mask[max_index % 3, int(row * grid_w[level] + col)] = 1
        yolo_1_all = [yolo_1, yolo_1_mask, true_bboxs[2]]
        yolo_2_all = [yolo_2, yolo_2_mask, true_bboxs[1]]
        yolo_3_all = [yolo_3, yolo_3_mask, true_bboxs[0]]
        return yolo_1_all, yolo_2_all, yolo_3_all

    def parse_label(self):
        img_names = []
        img_bboxs = []
        f = open(self.label_path, 'r')
        lines = f.readlines()
        coords = []
        is_First = True
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                img_path = line[2:]
                img_names.append(img_path)
                if is_First:
                    is_First = False
                else:
                    img_bboxs.append(coords.copy())
                    coords.clear()
            else:
                line = line.split(' ')
                x_min, y_min, w, h = float(line[0]), float(line[1]), float(line[2]), float(line[3])
                # [x1, y1, x2, y2, class_index]
                # for widerface, set face as 1, background as 0
                coord = [x_min, y_min, x_min + w, y_min + h, 1]
                coords.append(coord)
        # process last image
        img_bboxs.append(coords.copy())
        return img_names, img_bboxs
