"""generate anchors
    # Adapted from Ngoc Anh Huynh:
    # https://github.com/experiencor/keras-yolo3/gen_anchors.py
"""
import sys
import random
import argparse
import numpy as np
from tools.utils import parse_wider_annotation, parse_voc_annotation
from tools.voc_data import labels
sys.path.append("..")
sys.path.append("../tools")

parser = argparse.ArgumentParser("--------gen_anchors--------")
parser.add_argument('--annos_root', default='/Users/linyang/Desktop/data/WIDER_train/label.txt', type=str, help='voc: annotations xml dir, widerface: label.txt path')
parser.add_argument('--imgs_root', default='/Users/linyang/Desktop/data/WIDER_train/images/', type=str, help='images dir')
parser.add_argument('--data_type', default='wider', type=str, help="voc for VOC xml, wider for WiderFace")
parser.add_argument('--anchor_num', default=9, type=int, help="anchor numbers")
args = parser.parse_args()


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def avg_IOU(anns, centroids):
    n, d = anns.shape
    sum = 0.
    for i in range(anns.shape[0]):
        sum += max(IOU(anns[i], centroids))
    return sum / n


def print_anchors(centroids):
    anchors = centroids.copy()
    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    out_string = "anchors(width, height): ["
    for i in sorted_indices:
        out_string += "[" + str(int(anchors[i, 0] * 416)) + ', ' + str(int(anchors[i, 1] * 416)) + '], '
    out_string = out_string[:-2] + "]"
    print(out_string)


def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.ones(ann_num) * (-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))
    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances)  # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances - distances))))

        # assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def gen_anchors(annos_path, imgs_path, num_anchors, data_type):
    if data_type == "voc":
        all_insts  = parse_voc_annotation(annos_path, imgs_path, labels)
    else:
        all_insts = parse_wider_annotation(annos_path, imgs_path)
    # run k_mean to find the anchors
    annotation_dims = []
    for inst in all_insts:
        print(inst['filename'])
        for obj in inst['object']:
            relative_w = (float(obj['xmax']) - float(obj['xmin'])) / inst['width']
            relatice_h = (float(obj["ymax"]) - float(obj['ymin'])) / inst['height']
            annotation_dims.append(tuple(map(float, (relative_w, relatice_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print('\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids))
    print_anchors(centroids)


if __name__ == '__main__':
    annos_path = args.annos_root
    imgs_path = args.imgs_root
    anchor_num = args.anchor_num
    data_type = args.data_type
    gen_anchors(annos_path, imgs_path, anchor_num, data_type)