import base64
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
# 这个代码实现斑马鱼头背部皮肤的分割，来制作新的back训练数据
path0 = 'D:/00a/singlesickfish/sick/saveimages/'
piclist = os.listdir(path0)
import json
for pic in piclist:
    if pic.endswith('jpg'):
        pass
    else:
        continue
    picpath = path0 + pic
    # 获取图像的大小
    image = Image.open(picpath)
    image_width, image_height = image.size
    # with open(picpath,"rb") as image_file:
    #     image_data = base64.b64encode(image_file.read()).decode('utf-8')
    # 构建Labelme标注的JSON格式
    labelme_json = {
        "version": "5.0.5",
        "flags": {},
        "shapes": [],
        "imagePath": pic,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    # # 模型路径   暂时不要两点
    # # model = YOLO("D:/ultralytics-main/runs/pose/train12/weights/best.pt")  # 关键点模型   这个是原来一千多张wt斑马鱼训练出来的鱼背关键点识别模型
    # model = YOLO("D:/ultralytics-main/runs/pose/train/weights/best.pt")  # 关键点模型    这是czwt一共四千九百多张鱼背训练处理的关键点识别模型
    # # 如果保存的话：
    # results = model(source=picpath, save=False, show_labels=True, show_conf=True, boxes=True, conf=0.7)
    # for result in results:
    #     twokeypoints = result.keypoints.xy.tolist()
    #     print(twokeypoints)
    #     point1 = twokeypoints[0][0]
    #     point2 = twokeypoints[0][1]
    #     print(point1)
    #     print(point2)
    #     for i in range(0,2):
    #         if i == 0:
    #             labelme_json["shapes"].append({
    #                 "label": "1",
    #                 "points": [point1],
    #                 "group_id": None,
    #                 "shape_type": "point",
    #                 "flags": {}
    #             })
    #         else:
    #             labelme_json["shapes"].append({
    #                 "label": "2",
    #                 "points": [point2],
    #                 "group_id": None,
    #                 "shape_type": "polygon",
    #                 "flags": {}
    #             })

    # 模型路径
    # model = YOLO("D:/ultralytics-main/runs/segment/train3/weights/best.pt")  # 鱼背的分割模型    这是原来一千多张wt训练出来的鱼背分割模型
    # model = YOLO("D:/ultralytics-main/runs/segment/train11/weights/best.pt")  # 鱼背的分割模型    这是czwt一共四千多张鱼背训练出来的鱼背分割模型
    model = YOLO("D:/ultralytics-main/runs/segment/train2/weights/best.pt")   # 鱼体分割模型
    # 如果保存的话：
    results = model(source=picpath, save=False, show_labels=True, show_conf=True, boxes=True, conf=0.7)
    for result in results:
        if result.masks is not None and len(result.masks) > 0:
            print(result.masks.xy)
            arrs = result.masks.xy

            for arr in arrs:
                points = []
                a = []
                i = 0
                for x in np.nditer(arr):
                    a.append(float(x))
                    if i % 2 == 1:
                        points.append(a)
                        a = []
                    i = i + 1
                print(points)
                labelme_json["shapes"].append({
                    "label": "back",
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })

    # 将Labelme标注输出写入文件
    jsonpath = picpath.split(".")[0] + ".json"
    with open(jsonpath, "w") as json_file:
        json.dump(labelme_json, json_file, indent=4)

