
import glob
from PIL import Image
from ultralytics import YOLO
import csv
import os
from os.path import join , basename
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import math
# 这个代码实现了斑马鱼头背部皮肤的旋转对齐和分割，可以得到back数据集的制作

# 12.16 back背部图案已经进行了旋转对齐和分割，已经可以直接来进行back数据集的制作

# 两点的识别在train12
# 鱼背的识别需要先两点识别，旋转对齐之后再进行分割

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#path0 = 'D:/yolov8datasets/0backdataset/K/0730/'
path0 = 'D:/yolov8datasets/1/back50test2/'
# 这里遍历目录下的子目录
dirlist = os.listdir(path0)
for dir in dirlist:     # 这里的dir是D1、D2、等
    path = path0 + dir + '/'
    delldir = os.listdir(path0+dir)
    for dellfile in delldir:
        if dellfile=='back':
            # 图片路径
            source = path + dellfile

            # 预测图片的保存目录
            pred_dir = path+'1duiqi/'     # 对关键点识别之后的图片进行旋转
            mkdir(pred_dir)

            # 识别后的原始图片
            rcpic = path+"cdysave/"    # 关键点识别之后的原始图片

            # 模型路径
            # model = YOLO("D:/ultralytics-main/runs/pose/train12/weights/best.pt")  # 关键点模型   这个是原来一千多张wt斑马鱼训练出来的鱼背关键点识别模型
            model = YOLO("D:/ultralytics-main/runs/pose/train/weights/best.pt")  # 关键点模型    这是czwt一共四千九百多张鱼背训练处理的关键点识别模型
            # 如果保存的话：
            results = model(source=source,save=True,name=rcpic,show_labels=True,show_conf=True,boxes=True,conf=0.7)
            for result in results:
                image_name = basename(result.path)
                print(image_name)
                # print(result.keypoints.conf)
                if result.keypoints.conf != None :
                    conflist = result.keypoints.conf[0].tolist()
                    # print(conflist)
                    confflag = -1
                    for i in conflist:
                        if i < 0.5:
                            confflag = 0
                            break
                    if confflag == -1:
                        orig_img = result.orig_img   # 原始图像的矩阵
                        twokeypoints = result.keypoints.xy.tolist()
                        point1 = twokeypoints[0][0]
                        point2 = twokeypoints[0][1]
                        # print(point1)
                        # print(point2)
                        # 计算1与2两点连线与水平方向的夹角angle   夹角的形成：起始点1，结束点2, 12连线，从点2向左画一条水平线，与连线12形成的夹角
                        dy = point2[1] - point1[1]
                        dx = point2[0] - point1[0]
                        angle = math.atan2(dy,dx) * 180. / math.pi
                        heightNew = int(orig_img.shape[1] * math.fabs(math.sin(math.radians(angle))) + orig_img.shape[
                            0] * math.fabs(
                            math.cos(math.radians(angle))))
                        widthNew = int(orig_img.shape[0] * math.fabs(math.sin(math.radians(angle))) + orig_img.shape[
                            1] * math.fabs(
                            math.cos(math.radians(angle))))
                        # cv2.getRotationMatrix2D(center, angle, scale)
                        # center：旋转中心坐标，是一个元组参数(col, row)
                        # angle：旋转角度，旋转方向，负号为逆时针，正号为顺时针
                        # scale：旋转后图像相比原来的缩放比例，1为等比例缩放
                        rotate_matrix = cv2.getRotationMatrix2D((orig_img.shape[1] / 2, orig_img.shape[0] / 2), angle - 90, scale=1)
                        rotate_matrix[0, 2] += (widthNew - orig_img.shape[1]) / 2
                        rotate_matrix[1, 2] += (heightNew - orig_img.shape[0]) / 2
                        rotated_img = cv2.warpAffine(orig_img, rotate_matrix, (widthNew, heightNew))
                        # print(rotated_img)
                        cv2.imwrite(pred_dir + image_name, rotated_img)
            # 下面开始分割了
            # 需要进行背景分割的图像目录
            source = pred_dir
            # 分割之后保存的图像
            segdata = path + '1seg/'
            # 提取的分割图像
            saveseg = path + 'segdata/'
            mkdir(saveseg)
            # 模型路径
            # model = YOLO("D:/ultralytics-main/runs/segment/train3/weights/best.pt")  # 鱼背的分割模型    这是原来一千多张wt训练出来的鱼背分割模型
            # model = YOLO("D:/ultralytics-main/runs/segment/train11/weights/best.pt")  # 鱼背的分割模型    这是czwt一共四千多张鱼背训练出来的鱼背分割模型
            model = YOLO("D:/ultralytics-main/runs/segment/train5/weights/best.pt")  # 鱼背的分割模型    这是czwt一共六千多张鱼背训练出来的鱼背分割模型
            # 如果保存的话：
            results = model(source=source, save=True,name=segdata, show_labels=True, show_conf=True, boxes=True, conf=0.7)

            for result in results:
                image_name = basename(result.path)  # 提取图片名称
                # print(result.boxes.conf)
                mask_name = f"{os.path.splitext(image_name)[0]}.jpg"  # 根据图片名称生成保存结果的名称
                maskdir = path + '/savemask'
                mkdir(maskdir)
                pred_image_path = join(maskdir, mask_name)  # mask图片保存路径
                # 检测到鱼体时：
                if result.masks is not None and len(result.masks) > 0:
                    # print("输出x1,y1;x2,y2:")
                    # print(result.boxes.xyxy.tolist())
                    # print(result.boxes.conf)
                    xyxy = result.boxes.xyxy.tolist()[0]
                    # print(xyxy)
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = math.ceil(xyxy[2])
                    y2 = math.ceil(xyxy[3])

                    masks_data = result.masks.data
                    for index, mask in enumerate(masks_data):
                        mask = mask.cpu().numpy() * 255
                        cv2.imwrite(pred_image_path, mask)  # 保存mask图片
                        ori = cv2.imread(result.path)
                        mask = cv2.imread(pred_image_path, cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, (ori.shape[1], ori.shape[0]))
                        mask = mask / 255.0
                        ori[:, :, 0] = ori[:, :, 0] * mask
                        ori[:, :, 1] = ori[:, :, 1] * mask
                        ori[:, :, 2] = ori[:, :, 2] * mask
                        corpImg = ori[y1:y2, x1:x2]  # 裁剪图片试试  裁出来长方形
                        cv2.imwrite(saveseg + image_name, corpImg)





