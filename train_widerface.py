import math
import time
import datetime
import argparse
import os
import torch
import random
from vision.yolo_resnet import Yolo_V3
from vision.loss import YoloLossLayer
from tools.widerface_data import WiderFaceDataset, labels, anchors
from tools.augment import DetectionAugmentation
from torch.utils.data import DataLoader
from torchvision import transforms
from tools.utils import adjust_learning_rate


parser = argparse.ArgumentParser("--------Train YOLO-V3 WiderFace DataSet--------")
parser.add_argument('--weights_save_folder', default='./weights', type=str, help='Dir to save weights')
parser.add_argument('--imgs_root', default='data/WIDER_train/images/', help='train images dir')
parser.add_argument('--label_path', default='data/WIDER_train/label.txt', type=str, help='annotation xml dir')
parser.add_argument('--batch_size', default=16, type=int, help="batch size")
parser.add_argument('--net_w', default=416, type=int, help="input image width")
parser.add_argument('--net_h', default=416, type=int, help="input image height")
parser.add_argument('--grid_scale', default=[3, 1, 1], type=list, help="loss weights for yolo-v3 three scale")
parser.add_argument('--multi_scale', default=False, type=bool, help="whether use multi scale image to train")
parser.add_argument('--max_epoch', default=70, type=int, help="max training epoch")
parser.add_argument('--initial_lr', default=1e-3, type=float, help="initial learning rate")
parser.add_argument('--gamma', default=0.1, type=float, help="gamma for adjust lr")
parser.add_argument('--weight_decay', default=5e-4, type=float, help="weights decay")
parser.add_argument('--decay1', default=20, type=int)
parser.add_argument('--decay2', default=50, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--num_gpu', default=1, type=int, help="gpu number")
parser.add_argument('--pre_train', default=True, type=bool, help="whether use pre-train weights for change class number")
args = parser.parse_args()


def train(net, yolo_loss1, yolo_loss2, yolo_loss3, optimizer, trainSet, use_gpu):
    multi_scale = args.multi_scale
    epoch_size = math.ceil(len(trainSet) / args.batch_size)
    max_iter = args.max_epoch * epoch_size
    iteration = 0
    stepvalues = (args.decay1 * epoch_size, args.decay2 * epoch_size)
    step_index = 0
    dataLoader = DataLoader(trainSet, args.batch_size, shuffle=True, num_workers=args.num_workers)
    grid_scale = args.grid_scale
    print("Begin training...")
    for epoch in range(args.max_epoch):
        net.train()

        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(args.initial_lr, optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        if epoch % 10 == 0 and epoch > 0:

            if args.num_gpu > 1:
                torch.save(net.module.state_dict(),
                           os.path.join(args.weights_save_folder, 'epoch_' + str(epoch) + '.pth'))
            else:
                torch.save(net.state_dict(), os.path.join(args.weights_save_folder, 'epoch_' + str(epoch) + '.pth'))

        if iteration % 10 == 0 and iteration > 0 and multi_scale:
            # multi-scale, default [320, 416, 480, 640], must be divisible by 32
            # When the number of worker is not 0, there will be some batch_size cache, and the returned GT will not
            # change size immediately.
            scales = [320, 416, 480, 640]
            idx = random.randint(0, len(scales) - 1)
            trainSet.target_size = (scales[idx], scales[idx])
            print("use multi scale training, change image size to ({}, {})".format(scales[idx], scales[idx]))

        for ind, (imgs, yolo_l1, yolo_l2, yolo_l3) in enumerate(dataLoader):

            iteration += 1
            load_t0 = time.time()
            if use_gpu:
                imgs = imgs.cuda()

            net_out = net(imgs)
            optimizer.zero_grad()
            loss_scale1 = yolo_loss1(net_out[0], yolo_l1) * grid_scale[0]
            loss_scale2 = yolo_loss2(net_out[1], yolo_l2) * grid_scale[1]
            loss_scale3 = yolo_loss3(net_out[2], yolo_l3) * grid_scale[2]

            total_loss = loss_scale1 + loss_scale2 + loss_scale3
            total_loss.backward()
            optimizer.step()
            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            print(
                'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loss: {:.4f}|| LR: {:.8f} || Batchtime: {:.4f} s ||'
                ' ETA: {}'.format(epoch, args.max_epoch, (iteration % epoch_size) + 1, epoch_size, iteration + 1,
                                  max_iter, total_loss, lr, batch_time, str(datetime.timedelta(seconds=eta))))

    if args.num_gpu > 1:
        torch.save(net.module.state_dict(), os.path.join(args.weights_save_folder, 'face.pth'))
    else:
        torch.save(net.state_dict(), os.path.join(args.weights_save_folder, 'face.pth'))
    print('Finished Training')


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    net = Yolo_V3(class_num=len(labels), anchors=[3, 3, 3])

    if args.pre_train:
        device = torch.device("cuda" if use_gpu else "cpu")
        pretrained_dict = torch.load(os.path.join(args.weights_save_folder, "face.pth"), map_location=torch.device(device))
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    if args.num_gpu > 1 and use_gpu:
        net = torch.nn.DataParallel(net).cuda()
    elif use_gpu:
        net = net.cuda()

    if not os.path.exists(args.weights_save_folder):
        os.mkdir(args.weights_save_folder)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    augmentation = DetectionAugmentation()
    trainSet = WiderFaceDataset(img_root=args.imgs_root,
                                label_path=args.label_path,
                                target_size=(416, 416),
                                name_list=labels,
                                anchors=anchors,
                                transform=transform)
    yolo_loss1 = YoloLossLayer(anchors=anchors[6:], class_number=len(labels), reduction=32, use_gpu=use_gpu)
    yolo_loss2 = YoloLossLayer(anchors=anchors[3:6], class_number=len(labels), reduction=16, use_gpu=use_gpu)
    yolo_loss3 = YoloLossLayer(anchors=anchors[:3], class_number=len(labels), reduction=8, use_gpu=use_gpu)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.initial_lr, weight_decay=args.weight_decay)
    train(net, yolo_loss1, yolo_loss2, yolo_loss3, optimizer, trainSet, use_gpu)
