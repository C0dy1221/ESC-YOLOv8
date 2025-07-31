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
# 该程序对分帧之后的视频进行分割旋转对齐，生成人工筛选一遍就可以用来做个体识别的数据集  斑马鱼鱼体数据集的制作
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# path0 = 'D:/01grasscarpleftandright/0615AB/'
path0 = 'D:/01grasscarpleftandright/0425/'

# 这里遍历目录下的子目录
dirlist = os.listdir(path0)
for dir in dirlist:     # dir 就是A几B几
    print(dir)
    path = path0 + dir + '/'
    delldir = os.listdir(path0+dir)
    for dellfile in delldir:
        if 'C' or 'E' or 'F' or 'G' or 'H' in dellfile:
            source = path + dellfile
            # 图片路径
            # source = path + '/J1_TimeLapse'    # 需要进行背景分割的图像目录
            # 预测图片的保存目录
            pred_dir = path+'1seg/'     # 对分割并裁剪后的图像进行保存

            mkdir(pred_dir)
            # 模型路径
            model = YOLO("D:/ultralytics-main/runs/segment/train2/weights/best.pt")  # 分割模型   train2里面的模型是所有鱼体分割数据训练出来的最好的模型，
                                                                                     #           train1里面是所有WT的鱼体分割模型训练出来的模型
            # 如果保存的话：
            results = model(source=source,save=False,show_labels=True,show_conf=True,boxes=True,conf=0.9)

            for result in results:
                image_name = basename(result.path)  # 提取图片名称
                # print(result.boxes.conf)
                mask_name = f"{os.path.splitext(image_name)[0]}.jpg"  # 根据图片名称生成保存结果的名称
                maskdir = path+'/savemask'
                mkdir(maskdir)
                pred_image_path = join(maskdir, mask_name) # mask图片保存路径
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
                        cv2.imwrite(pred_image_path , mask)   # 保存mask图片
                        ori = cv2.imread(result.path)
                        mask = cv2.imread(pred_image_path, cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, (ori.shape[1], ori.shape[0]))
                        mask = mask / 255.0
                        ori[:, :, 0] = ori[:, :, 0] * mask
                        ori[:, :, 1] = ori[:, :, 1] * mask
                        ori[:, :, 2] = ori[:, :, 2] * mask
                        corpImg = ori[y1:y2, x1:x2]  # 裁剪图片试试  裁出来长方形
                        cv2.imwrite(pred_dir + image_name, corpImg)


            # 模型路径
            model = YOLO("D:/ultralytics-main/runs/pose/train11/weights/best.pt")   # 关键点识别旋转对齐
            # 图片路径
            source = pred_dir    # 需要旋转的原始图片
            # 预测图片的保存目录
            pred_dir_leftandright = path + '/align/'
            mkdir(pred_dir_leftandright)

            # 如果保存的话：
            results = model(source=source,save=False,show_labels=True,show_conf=True,boxes=True,conf=0.9)
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
                        # 计算鱼眼1与鱼尾柄2上两点连线与水平方向的夹角angle   夹角的形成：起始点1，结束点2, 12连线，从点2向左画一条水平线，与连线12形成的夹角
                        dy = point2[1] - point1[1]
                        dx = point2[0] - point1[0]
                        angle = math.atan2(dy,dx) * 180. /math.pi
                        # print("输出angle:")
                        # print(angle)
                        # print(orig_img.shape)
                        # 获取旋转矩阵
                        # 参数1为旋转中心点;
                        # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
                        # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
                        # cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
                        if angle>90:    # angle大于0，鱼头在右上
                            heightNew = int(orig_img.shape[1] * math.fabs(math.sin(math.radians(angle))) + orig_img.shape[0] * math.fabs(
                                math.cos(math.radians(angle))))
                            widthNew = int(orig_img.shape[0] * math.fabs(math.sin(math.radians(angle))) + orig_img.shape[1] * math.fabs(
                                math.cos(math.radians(angle))))
                            rotate_matrix = cv2.getRotationMatrix2D((orig_img.shape[1]/2, orig_img.shape[0]/2),angle-180,scale=1)
                            rotate_matrix[0, 2] += (widthNew - orig_img.shape[1]) / 2
                            rotate_matrix[1, 2] += (heightNew - orig_img.shape[0]) / 2
                            rotated_img = cv2.warpAffine(orig_img,rotate_matrix,(widthNew, heightNew))
                            # print(rotated_img)
                            cv2.imwrite(pred_dir_leftandright+image_name,rotated_img)
                        elif angle < -90:   # 鱼头在右下
                            heightNew = int(orig_img.shape[1] * math.fabs(math.sin(math.radians(angle))) + orig_img.shape[0] * math.fabs(
                                math.cos(math.radians(angle))))
                            widthNew = int(orig_img.shape[0] * math.fabs(math.sin(math.radians(angle))) + orig_img.shape[1] * math.fabs(
                                math.cos(math.radians(angle))))
                            rotate_matrix = cv2.getRotationMatrix2D((orig_img.shape[1]/2, orig_img.shape[0]/2), angle + 180, scale=1)
                            rotate_matrix[0, 2] += (widthNew - orig_img.shape[1]) / 2
                            rotate_matrix[1, 2] += (heightNew - orig_img.shape[0]) / 2
                            rotated_img = cv2.warpAffine(orig_img, rotate_matrix, (widthNew, heightNew))
                            # print(rotated_img)
                            cv2.imwrite(pred_dir_leftandright + image_name, rotated_img)
                        elif angle >=0 and angle <=90:    # 鱼头在左上
                            heightNew = int(orig_img.shape[1] * math.fabs(math.sin(math.radians(angle))) + orig_img.shape[0] * math.fabs(
                                math.cos(math.radians(angle))))
                            widthNew = int(orig_img.shape[0] * math.fabs(math.sin(math.radians(angle))) + orig_img.shape[1] * math.fabs(
                                math.cos(math.radians(angle))))
                            rotate_matrix = cv2.getRotationMatrix2D((orig_img.shape[1]/2, orig_img.shape[0]/2), angle, scale=1)
                            rotate_matrix[0, 2] += (widthNew - orig_img.shape[1]) / 2
                            rotate_matrix[1, 2] += (heightNew - orig_img.shape[0]) / 2
                            rotated_img = cv2.warpAffine(orig_img, rotate_matrix, (widthNew, heightNew))
                            # print(rotated_img)
                            cv2.imwrite(pred_dir_leftandright + image_name, rotated_img)
                        elif angle<0 and angle >= -90:   # 鱼头在左下
                            heightNew = int(orig_img.shape[1] * math.fabs(math.sin(math.radians(angle))) + orig_img.shape[0] * math.fabs(
                                math.cos(math.radians(angle))))
                            widthNew = int(orig_img.shape[0] * math.fabs(math.sin(math.radians(angle))) + orig_img.shape[1] * math.fabs(
                                math.cos(math.radians(angle))))
                            rotate_matrix = cv2.getRotationMatrix2D((orig_img.shape[1]/2, orig_img.shape[0]/2), angle, scale=1)
                            rotate_matrix[0, 2] += (widthNew - orig_img.shape[1]) / 2
                            rotate_matrix[1, 2] += (heightNew - orig_img.shape[0]) / 2
                            rotated_img = cv2.warpAffine(orig_img, rotate_matrix, (widthNew, heightNew))
                            # print(rotated_img)
                            cv2.imwrite(pred_dir_leftandright + image_name, rotated_img)

            # 模型路径
            model = YOLO("D:/ultralytics-main/runs/pose/train11/weights/best.pt")
            # 图片路径
            source = pred_dir_leftandright   # 需要corp的图片
            pred_dir_left = path+'/saveleft/'    # 旋转后鱼头向左
            pred_dir_right = path+'/saveright/'   # 旋转后鱼头向右
            mkdir(pred_dir_left)
            mkdir(pred_dir_right)
            # 如果保存的话：
            results = model(source=source,save=False,show_labels=True,show_conf=True,boxes=True)
            for result in results:
                image_name = basename(result.path)
                ori = cv2.imread(result.path)
                orig_img = result.orig_img   # 原始图像的矩阵
                print(image_name)
                if len(result.boxes.xyxy.tolist()) == 0:
                    continue
                # print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
                # print(result.boxes.xyxy.tolist())
                xyxy = result.boxes.xyxy.tolist()[0]
                # print(xyxy)
                x1 = int(xyxy[0])
                y1 = int(xyxy[1])
                x2 = math.ceil(xyxy[2])
                y2 = math.ceil(xyxy[3])
                if len(result.keypoints.xy.tolist()) == 0:
                    continue
                twokeypoints = result.keypoints.xy.tolist()
                point1 = twokeypoints[0][0]
                point2 = twokeypoints[0][1]
                print(point1)
                print(point2)
                corpImg = ori[y1:y2, x1:x2]  # 裁剪图片试试  裁出来长方形
                if point1[0]<=point2[0]:
                    cv2.imwrite(pred_dir_left + image_name, corpImg)
                else:
                    cv2.imwrite(pred_dir_right + image_name, corpImg)