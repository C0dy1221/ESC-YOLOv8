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

# 该程序对斑马鱼背部图像进行分割

# 模型路径
model = YOLO("D:/ultralytics-main/runs/segment/train3/weights/best.pt")    # 鱼背分割模型的地址
# 图片路径
source = 'D:/yolov8datasets/backtest/source/'
# 预测图片的保存目录
pred_dir = 'D:/yolov8datasets/backtest/save/'


# 如果保存的话：
results = model(source=source,save=True,name=pred_dir,show_labels=True,show_conf=True,boxes=True)

# 如果不保存的话：
# results = model(source=source,show_labels=False,show_conf=False,boxes=False)

for result in results:
    image_name = basename(result.path)  # 提取图片名称
    mask_name = f"{os.path.splitext(image_name)[0]}.jpg"  # 根据图片名称生成保存结果的名称
    pred_image_path = join('D:/yolov8datasets/backtest/savemask', mask_name)# 图片保存路径
    # 检测到鱼体时：
    if result.masks is not None and len(result.masks) > 0:
        masks_data = result.masks.data
        for index, mask in enumerate(masks_data):
            mask = mask.cpu().numpy() * 255
            # cv2.imwrite(f'./output_{index}.png', mask)
            cv2.imwrite(pred_image_path , mask)

maskspath = "D:/yolov8datasets/backtest/savemask/"
imgpath = os.listdir(source)
for img in imgpath:
    oriimg = source+"/"+img
    orimask = maskspath + img
    # print(oriimg)
    ori = cv2.imread(oriimg)
    mask = cv2.imread(orimask,cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(ori.shape[1],ori.shape[0]))
    mask = mask / 255.0
    ori[:, :, 0] = ori[:, :, 0] * mask
    ori[:, :, 1] = ori[:, :, 1] * mask
    ori[:, :, 2] = ori[:, :, 2] * mask
    sa = "D:/yolov8datasets/backtest/cdysave/"
    cv2.imwrite(sa+img,ori)
