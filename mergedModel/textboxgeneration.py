"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
# parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
# parser.add_argument('--test_folder', default='/test/', type=str, help='folder path to input images')

# args = parser.parse_args()
args = argparse.Namespace(canvas_size=1280, link_threshold=0.4, low_text=0.4, mag_ratio=1.5, poly=False,
                          show_time=False, text_threshold=0.7, trained_model='craft_mlt_25k.pth')
# print(args)


result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def test_net(net, image, text_threshold, link_threshold, low_text, poly):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]

    if torch.cuda.is_available():
        x = x.cuda()

    # forward pass
    #print(time.time())
    y, _ = net(x)
    #print(time.time())

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Post-processing
    boxes = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    # polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    # for k in range(len(polys)):
    # if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    # render_img = score_text.copy()
    # render_img = np.hstack((render_img, score_link))
    # ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes


def loadmodel():
    # load net
    net = CRAFT()  # initialize

    # print('Loading weights from checkpoint (' + args.trained_model + ')')
    if torch.cuda.is_available():
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
        # cudnn.deterministic = True
    else:
        net = net.cpu()
        net = torch.nn.DataParallel(net)

    net.eval()

    return net


def get_text_locs(net, path):
    """ For test images in a folder """
    image = imgproc.loadImage(path)
    with torch.no_grad():
        polys = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.poly)

    poly = []
    for i, box in enumerate(polys):
        poly.append(np.array(box).astype(np.int32).reshape((-1)).tolist())

    return poly
