import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="none"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, x, target):
        loss = self.__loss(input=x, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(x)), self.__gamma)

        return loss


class YoloLossLayer(nn.Module):
    def __init__(self, anchors, class_number, reduction, coord_scale=5.0, noobj_scale=1,
                 obj_scale=5, class_scale=1.0, obj_thresh=0.5, net_factor=(416, 416), neg_mining=True, negpos_ratio=7, use_gpu=False):
        super(YoloLossLayer, self).__init__()
        self.anchor_number = len(anchors)
        self.anchors = torch.Tensor(anchors)
        self.class_number = class_number
        self.reduction = reduction
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale
        self.obj_thresh = obj_thresh
        self.neg_mining = neg_mining
        self.negpos_ratio = negpos_ratio
        self.use_gpu = use_gpu
        self.net_factor = net_factor

    def forward(self, net_out, ground_truth):
        batch_size, grid_h, grid_w, out_channel = net_out.shape
        yolo, anchor_mask, true_bboxs = ground_truth

        if self.use_gpu:
            yolo = yolo.cuda()
            anchor_mask = anchor_mask.cuda()
            true_bboxs = true_bboxs.cuda()

        anchor_mask = anchor_mask.unsqueeze(dim=2)

        net_out = net_out.view(batch_size, self.anchor_number, self.class_number + 4 + 1, grid_h * grid_w)

        coords = torch.zeros_like(net_out[:, :, :4, :])
        if self.use_gpu:
            coords = coords.cuda()

        coords[:, :, :2, :] = net_out[:, :, :2, :].sigmoid()
        coords[:, :, 2:4, :] = net_out[:, :, 2:4, :]
        p_conf = net_out[:, :, 4, :].sigmoid()
        p_clas = net_out[:, :, 5:, :]

        # classification loss
        t_clas = yolo[:, :, 5:, :]
        clas_loss = F.binary_cross_entropy_with_logits(p_clas, t_clas, reduction='none') * anchor_mask

        # coords loss
        wh_loss_scale = 2.0 - 1.0 * yolo[:, :, 2:3, :] * yolo[:, :, 3:4, :] / (self.net_factor[0] * self.net_factor[1])
        xy_loss = F.smooth_l1_loss(coords[:, :, :2, :], yolo[:, :, :2, :], reduction='none') * anchor_mask
        wh_loss = F.smooth_l1_loss(coords[:, :, 2:4, :], yolo[:, :, 2:4, :], reduction='none') * anchor_mask * wh_loss_scale
        coords_loss = xy_loss.sum() + wh_loss.sum()

        # confidence loss
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")
        t_conf = yolo[:, :, 4, :]
        grid_predicts = self.rescale_to_img(coords, grid_h, grid_w).permute(0, 3, 1, 2).unsqueeze(3)
        iou_scores = self.compute_iou(grid_predicts, true_bboxs.unsqueeze(1).unsqueeze(1))
        iou_max = iou_scores.max(-1, keepdim=True)[0]
        noobj_mask = 1 - anchor_mask.squeeze(dim=2)
        label_noobj_mask = (iou_max < self.obj_thresh).squeeze(3).permute(0, 2, 1).float() * noobj_mask
        obj_conf_loss = anchor_mask.squeeze(dim=2) * FOCAL(x=p_conf, target=t_conf)
        noobj_conf_loss = label_noobj_mask * FOCAL(x=p_conf, target=t_conf)

        pos_count = anchor_mask.sum() + 1e-5
        neg_count = label_noobj_mask.sum() + 1e-5
        if self.neg_mining:
            noobj_conf_loss = noobj_conf_loss.view(1, -1)
            neg_count = torch.clamp(self.negpos_ratio * pos_count, max=neg_count - 1, min=20)
            conf_loss, _ = noobj_conf_loss.sort(1, descending=True)
            noobj_conf_loss = conf_loss[0, :neg_count.int()]

        clas_loss = clas_loss.sum() * self.class_scale / pos_count
        coords_loss = coords_loss * self.coord_scale / pos_count
        obj_conf_loss = obj_conf_loss.sum() * self.obj_scale / pos_count
        noobj_conf_loss = noobj_conf_loss.sum() * self.noobj_scale / neg_count
        total_loss = clas_loss + coords_loss + obj_conf_loss + noobj_conf_loss

        print("network output shape: ({}, {}),total loss: {}, class loss: {}, coords loss: {}, obj_conf loss: {}, "
              "noobj_conf loss: {}".format(grid_w, grid_h, total_loss, clas_loss, coords_loss, obj_conf_loss,
                                           noobj_conf_loss))

        return total_loss

    def rescale_to_img(self, coords, grid_h, grid_w):
        col_index = torch.arange(0, grid_w).repeat(grid_h, 1).view(grid_h * grid_w)
        row_index = torch.arange(0, grid_h).repeat(grid_w, 1).t().contiguous().view(grid_h * grid_h)
        img_coords = torch.zeros_like(coords)
        anchor_w = self.anchors[:, 0].contiguous().view(self.anchor_number, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.anchor_number, 1)
        if self.use_gpu:
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()
            col_index = col_index.cuda()
            row_index = row_index.cuda()
            img_coords = img_coords.cuda()
        # to img coords
        img_coords[:, :, 0, :] = (coords[:, :, 0, :] + col_index.float()) / grid_w * self.net_factor[0]
        img_coords[:, :, 1, :] = (coords[:, :, 1, :] + row_index.float()) / grid_h * self.net_factor[1]
        img_coords[:, :, 2, :] = coords[:, :, 2, :].exp() * anchor_w
        img_coords[:, :, 3, :] = coords[:, :, 3, :].exp() * anchor_h

        # to [x1, y1, x2, y2]
        img_coords[:, :, 0, :] = img_coords[:, :, 0, :] - img_coords[:, :, 2, :] / 2
        img_coords[:, :, 1, :] = img_coords[:, :, 1, :] - img_coords[:, :, 3, :] / 2
        img_coords[:, :, 2, :] = img_coords[:, :, 0, :] + img_coords[:, :, 2, :]
        img_coords[:, :, 3, :] = img_coords[:, :, 1, :] + img_coords[:, :, 3, :]
        return img_coords

    def compute_iou(self, boxes1, boxes2):
        tl = torch.max(boxes1[..., :2], boxes2[..., :2])
        br = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        wh = br - tl
        wh = torch.max(wh, torch.zeros_like(wh))
        inter = wh[..., 0] * wh[..., 1]
        area_1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area_2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        return inter / (area_1 + area_2 - inter + 1e-5)


# from torch.utils.data import DataLoader
# from torchvision import transforms
# from tools.widerface_data import WiderFaceDataset
# from tools.utils import custom_collate_fn
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
# data = WiderFaceDataset(img_root='/Users/linyang/Desktop/data/WIDER_train/images/', label_path="/Users/linyang/PycharmProjects/personal_projects/yolo-v3-pytorch/data/label.txt", target_size=(416, 416),
#                  anchors=[[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90],[90,100]], transform=transform)
# batch_iterator = DataLoader(data, 1, shuffle=False, num_workers=0)
# for i, (img, y1, y2, y3) in enumerate(batch_iterator):
#     net_out = torch.rand(1, 13, 13, 21)
#     yolo_loss = YoloLossLayer(anchors=[[30, 40],[50, 50],[70, 80]],class_number=2, reduction=32)
#     yolo_loss(net_out, y1)