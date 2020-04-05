import torch
import torch.nn as nn
import torch.nn.functional as F
from vision.resnet50 import ResNet


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0

        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1)

        self.merge1 = conv_bn(out_channels, out_channels)
        self.merge2 = conv_bn(out_channels, out_channels)

    def forward(self, input):
        input = list(input)

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return output1, output2, output3


class YoloHead(nn.Module):
    def __init__(self, class_num, inchannels, num_anchors):
        super(YoloHead, self).__init__()
        self.class_num = class_num
        self.num_anchors = num_anchors
        self.out_channels = self.num_anchors * (4 + 1 + self.class_num)
        self.conv = conv_bn(inchannels, self.out_channels)
        self.conv1x1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv(x)
        out = self.conv1x1(out)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out


class Yolo_V3(nn.Module):
    def __init__(self, class_num, anchors, pretrain=True):
        super(Yolo_V3, self).__init__()
        self.class_num = class_num
        self.anchors = anchors
        self.body = ResNet()
        self.pretrain = pretrain
        if self.pretrain:

            import torchvision.models.resnet as resnet
            pre_net = resnet.resnet50(pretrained=True)
            pretrained_dict = pre_net.state_dict()
            model_dict = self.body.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.body.load_state_dict(model_dict)

            for param in self.body.parameters():
                param.requires_grad = False

        in_channels_stage2 = 512
        in_channels_list = [
            in_channels_stage2 * 1,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
        ]
        out_channels = 64
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)

        self.yolo_head = self._make_yolo_head(fpn_num=3, inchannels=out_channels)

    def _make_yolo_head(self, fpn_num, inchannels):
        yolo_head = nn.ModuleList()
        for i in range(fpn_num):
            yolo_head.append(YoloHead(self.class_num, inchannels, self.anchors[i]))
        return yolo_head

    def forward(self, x):
        out = self.body(x)
        fpn = self.fpn(out)
        fpn = list(fpn)
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = fpn[2]
        features = [feature3, feature2, feature1]

        yolos = [self.yolo_head[i](feature) for i, feature in enumerate(features)]
        return yolos
