"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

#-*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

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

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    # Base arguments
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type result')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='use link refiner for sentense-level dataset')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    args = parser.parse_args()


    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.test_folder)

    # ----- สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์ -----
    result_folder = './result/'
    vis_folder = os.path.join(result_folder, 'visualization/')
    rotated_folder = os.path.join(result_folder, 'rotated/')

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    if not os.path.isdir(vis_folder):
        os.mkdir(vis_folder)
    if not os.path.isdir(rotated_folder):
        os.mkdir(rotated_folder)
    # ----------------------------------------

    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path) # image is RGB

        image_for_vis = image[:,:,::-1].copy() # image_for_vis is BGR

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        
        final_polys = []
        angle = 0 # Default angle
        if len(polys) > 0:
            all_points = np.concatenate(polys, axis=0)
            rect = cv2.minAreaRect(all_points)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            final_polys.append(box)

            # ----- คำนวณมุมและจุดต่างๆ -----
            sorted_corners = sorted(box, key=lambda p: p[1], reverse=True)
            bottom_p1 = sorted_corners[0]
            bottom_p2 = sorted_corners[1]
            
            if bottom_p1[0] < bottom_p2[0]:
                origin_point = tuple(bottom_p1)
                bottom_right_corner = tuple(bottom_p2)
            else:
                origin_point = tuple(bottom_p2)
                bottom_right_corner = tuple(bottom_p1)
            
            dx = bottom_right_corner[0] - origin_point[0]
            dy = bottom_right_corner[1] - origin_point[1]
            angle = np.degrees(np.arctan2(-dy, dx))
            print(f"\nมุมของขอบล่างข้อความ: {angle:.2f} องศา")
            # -----------------------------

            # ----- วาดภาพอธิบายมุม -----
            cv2.circle(image_for_vis, origin_point, 5, (255, 100, 0), -1)
            cv2.line(image_for_vis, origin_point, (origin_point[0] + 100, origin_point[1]), (255, 100, 0), 2)
            cv2.line(image_for_vis, origin_point, bottom_right_corner, (0, 255, 255), 2)
            
            angle_rad_for_text = np.deg2rad(angle / 2)
            text_radius = 60
            text_x = origin_point[0] + int(text_radius * np.cos(angle_rad_for_text))
            text_y = origin_point[1] - int(text_radius * np.sin(angle_rad_for_text))
            cv2.putText(image_for_vis, f"{angle:.1f} deg", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # ---------------------------

        # ----- หมุนภาพต้นฉบับเพื่อแก้ไข -----
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        # ------------------------------------

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(vis_folder, "res_" + filename + '_mask.jpg')
        cv2.imwrite(mask_file, score_text)

        # 1. บันทึกภาพที่วาดเส้นอธิบายและไฟล์ txt ลงในโฟลเดอร์ 'visualization'
        file_utils.saveResult(image_path, image_for_vis, final_polys, dirname=vis_folder)

        # 2. บันทึกภาพที่หมุนแก้ไขแล้วลงในโฟลเดอร์ 'rotated'
        rotated_filename = os.path.join(rotated_folder, "rotated_" + filename + ".png")
        cv2.imwrite(rotated_filename, cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR))


    print("elapsed time : {}s".format(time.time() - t))