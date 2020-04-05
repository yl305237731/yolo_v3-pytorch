import torch
import torch.nn as nn
import torch.nn.functional as F


class YoloLossLayer(nn.Module):
    def __init__(self, anchors, class_number, reduction, coord_scale=1, noobj_scale=1, obj_scale=5, class_scale=1.0,
                 obj_thresh=0.5, neg_mining=False, negpos_ratio=7, use_gpu=False):
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

    def forward(self, net_out, ground_truth):
        batch_size, grid_h, grid_w, out_channel = net_out.shape
        yolo, obj_mask, true_bboxs = ground_truth

        if self.use_gpu:
            yolo = yolo.cuda()
            obj_mask = obj_mask.cuda()
            true_bboxs = true_bboxs.cuda()

        obj_mask = obj_mask.unsqueeze(dim=2)

        net_out = net_out.view(batch_size, self.anchor_number, self.class_number + 4 + 1, grid_h * grid_w)

        p_loc = torch.zeros_like(net_out[:, :, :4, :])
        if self.use_gpu:
            p_loc = p_loc.cuda()

        p_loc[:, :, :2, :] = net_out[:, :, :2, :].sigmoid()
        p_loc[:, :, 2:4, :] = net_out[:, :, 2:4, :]
        p_conf = net_out[:, :, 4, :].sigmoid()
        p_clas = net_out[:, :, 5:, :].sigmoid()

        # classification loss
        t_clas = yolo[:, :, 5:, :]
        clas_loss = F.binary_cross_entropy(p_clas, t_clas, reduction='none') * obj_mask

        # coords loss
        wh_loss_scale = 2.0 - 1.0 * yolo[:, :, 2:3, :] * yolo[:, :, 3:4, :] / (self.reduction * self.reduction * grid_w * grid_h)
        xy_loss = F.smooth_l1_loss(p_loc[:, :, :2, :], yolo[:, :, :2, :], reduction='none') * obj_mask * wh_loss_scale
        wh_loss = F.smooth_l1_loss(p_loc[:, :, 2:4, :], yolo[:, :, 2:4, :], reduction='none') * obj_mask * wh_loss_scale
        loc_loss = xy_loss.sum() + wh_loss.sum()

        # confidence loss
        t_conf = yolo[:, :, 4, :]
        grid_predicts = self.rescale_to_img(p_loc, grid_h, grid_w).permute(0, 3, 1, 2).unsqueeze(3)
        iou_scores = self.compute_iou(grid_predicts, true_bboxs.unsqueeze(1).unsqueeze(1))
        iou_max = iou_scores.max(-1, keepdim=True)[0]
        noobj_mask = 1 - obj_mask.squeeze(dim=2)
        label_noobj_mask = (iou_max < self.obj_thresh).squeeze(3).permute(0, 2, 1).float() * noobj_mask
        obj_conf_loss = obj_mask.squeeze(dim=2) * self.focal_loss(p_conf, t_conf, gamma=1)
        noobj_conf_loss = label_noobj_mask * self.focal_loss(p_conf, t_conf)

        pos_count = obj_mask.sum() + 1e-5
        neg_count = label_noobj_mask.sum() + 1e-5

        if self.neg_mining:
            neg_count = torch.clamp(self.negpos_ratio * pos_count, max=neg_count - 1)
            conf_loss, _ = noobj_conf_loss.sort(1, descending=True)
            noobj_conf_loss = conf_loss[0, :neg_count.int()]

        clas_loss = clas_loss.sum() * self.class_scale
        coords_loss = loc_loss * self.coord_scale
        obj_conf_loss = obj_conf_loss.sum() * self.obj_scale
        noobj_conf_loss = noobj_conf_loss.sum() * self.noobj_scale
        total_loss = clas_loss + coords_loss + obj_conf_loss + noobj_conf_loss
        # Online statistics
        print("network output shape: ({}, {}), pos_num: {}, neg_num: {}, total loss: {}, class loss: {}, coords loss: "
              "{}, obj_conf loss: {}, noobj_conf loss: {}".format(grid_w, grid_h, torch.floor(pos_count - 1),
                                                                  torch.floor(neg_count), total_loss, clas_loss,
                                                                  coords_loss, obj_conf_loss, noobj_conf_loss))
        print("object average conf: {}".format((obj_mask.squeeze(dim=2) * p_conf).sum() / (pos_count - 1)))
        print("background average conf: {}".format((label_noobj_mask * p_conf).mean()))
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
        img_coords[:, :, 0, :] = (coords[:, :, 0, :] + col_index.float()) / grid_w * (self.reduction * grid_w)
        img_coords[:, :, 1, :] = (coords[:, :, 1, :] + row_index.float()) / grid_h * (self.reduction * grid_h)
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

    def focal_loss(self, predict, target, alpha=1, gamma=2):
        bce_loss = F.binary_cross_entropy(predict, target, reduction='none')
        pt = torch.exp(-bce_loss)
        return alpha * (1 - pt) ** gamma * bce_loss
