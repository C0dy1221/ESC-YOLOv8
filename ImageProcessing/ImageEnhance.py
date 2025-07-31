# import cv2
# import numpy as np
# import glob
# from PIL import Image
# from ultralytics import YOLO
# import csv
# import os
# from os.path import join , basename
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
#
# import torch
# import torch.nn.functional as F
# import cv2
# import math
#
# def enhance_region(image, region):
#     """
#     Enhance the contrast of a specified region in the image.
#
#     :param image: Input image
#     :param region: Tuple (x, y, w, h) defining the region to enhance
#     :return: Image with enhanced region
#     """
#     x1, y1, x2, y2 = region
#     # Extract the region of interest (ROI)
#     roi = image[y1:y2, x1:x2]
#     hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#     # 增强图像的亮度和对比度
#     hsv_roi[:, :, 2] = cv2.equalizeHist(hsv_roi[:, :, 2])
#     # 将增强后的区域转换回BGR格式
#     enhanced_roi = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)
#
#     # 创建一个与原始图像相同的图像用于存储结果
#     enhanced_image = image.copy()
#     # 将增强后的区域复制回原始图像中的相应位置
#     enhanced_image[y1:y2, x1:x2] = enhanced_roi
#     # # Apply histogram equalization to the ROI
#     # enhanced_roi = cv2.equalizeHist(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
#
#     # # Merge the enhanced ROI back into the original image
#     # image[y1:y2, x1:x2 , 0] = enhanced_roi
#     # image[y1:y2, x1:x2 , 1] = enhanced_roi
#     # image[y1:y2, x1:x2 , 2] = enhanced_roi
#
#     return enhanced_image
#
#
# # Load an image from file
# image = cv2.imread('D:/EMA-yolov8-SPPLEAN/IMG20240416165432.jpg')
# source = 'D:/EMA-yolov8-SPPLEAN/IMG20240416165432.jpg'
# # 模型路径
# model = YOLO("D:/EMA-yolov8-SPPLEAN/runs/segment/train4/weights/best.pt")
# results = model(source=source,save=False,show_labels=True,show_conf=True,boxes=True)
# for result in results:
#     if result.masks is not None and len(result.masks) > 0:
#         # print(result.boxes)
#         all_xyxy = result.boxes.xyxy.tolist()
#         print(all_xyxy)
#         ii =0
#         for xyxy in all_xyxy:
#             ii = ii+1
#             x1 = int(xyxy[0])
#             y1 = int(xyxy[1])
#             x2 = math.ceil(xyxy[2])
#             y2 = math.ceil(xyxy[3])
#             region = (x1, y1, x2, y2)
#             if ii ==1:
#                 enhanced_image0 = enhance_region(image, region)
#                 cv2.imwrite("outimg.jpg",enhanced_image0)
#             else:
#                 enhanced_image0 = enhance_region(cv2.imread('outimg.jpg'), region)
#                 cv2.imwrite("outimg.jpg", enhanced_image0)
#
#         # print("ii")
#
#
#
# # Define the region to enhance (x, y, width, height)
#
#
# # Enhance the specified region
#
#
# # # Display the original and enhanced images
# # cv2.imshow('Original Image', image)
# # cv2.imshow('Enhanced Image', "outimg.jpg")
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # 读取彩色图像
# image = cv2.imread('D:/EMA-yolov8-SPPLEAN/IMG20240416165432.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # 1. 对比度增强（伽马校正）
# def adjust_gamma(image, gamma=1.0):
#     invGamma = 1.0 / gamma
#     table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
#     return cv2.LUT(image, table)
#
# gamma_corrected = adjust_gamma(image, gamma=1.5)
#
# # 2. 锐化
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# sharpened = cv2.filter2D(image, -1, kernel)
#
# # 3. 直方图均衡化（在HSV颜色空间进行）
# hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
# hist_equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#
# # 4. 自适应直方图均衡化（CLAHE，在LAB颜色空间进行）
# lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# lab[:, :, 0] = clahe.apply(lab[:, :, 0])
# clahe_equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
#
# # 显示图像
# titles = ['Original Image', 'Gamma Corrected', 'Sharpened', 'Histogram Equalized (HSV)', 'CLAHE Equalized (LAB)']
# images = [image, gamma_corrected, sharpened, hist_equalized, clahe_equalized]
#
# plt.figure(figsize=(12, 6))
# for i in range(5):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(cv2.GaussianBlur(images[i],(5,5),0))
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
#
# plt.tight_layout()
# plt.show()

# 小波变换去噪
# import cv2
# import pywt
# import numpy as np
# import matplotlib.pyplot as plt
#
# def wavelet_transform(image, wavelet='haar', level=2, threshold=50):
#     # 分解
#     coeffs = pywt.wavedec2(image, wavelet, level=level)
#     cA, cD = coeffs[0], coeffs[1:]
#
#     # 阈值处理
#     cD_processed = []
#     for details in cD:
#         cH, cV, cD = details
#         cH = pywt.threshold(cH, threshold, mode='soft')
#         cV = pywt.threshold(cV, threshold, mode='soft')
#         cD = pywt.threshold(cD, threshold, mode='soft')
#         cD_processed.append((cH, cV, cD))
#
#     # 重构
#     coeffs_processed = (cA, *cD_processed)
#     reconstructed_image = pywt.waverec2(coeffs_processed, wavelet)
#     return reconstructed_image
#
# # 读取彩色图像
# image = cv2.imread('D:/EMA-yolov8-SPPLEAN/IMG20240416165432.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # 分离颜色通道
# r_channel, g_channel, b_channel = cv2.split(image)
#
# # 对每个颜色通道进行小波变换和处理
# r_processed = wavelet_transform(r_channel)
# g_processed = wavelet_transform(g_channel)
# b_processed = wavelet_transform(b_channel)
#
# # 合并处理后的颜色通道
# processed_image = cv2.merge((r_processed, g_processed, b_processed))
# processed_image = np.uint8(processed_image)
#
# # 显示原图和处理后的图像
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image)
#
# plt.subplot(1, 2, 2)
# plt.title('Processed Image')
# plt.imshow(processed_image)
#
# plt.show()
#
# # 保存处理后的图像
# processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
# cv2.imwrite('processed_image.jpg', processed_image_bgr)

# 非局部均值滤波
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 读取彩色图像
# image = cv2.imread('D:/EMA-yolov8-SPPLEAN/IMG20240416165432.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # 添加噪声
# row, col, ch = image.shape
# mean = 0
# var = 0.01
# sigma = var ** 0.5
# gauss = np.random.normal(mean, sigma, (row, col, ch))
# noisy_image = image + gauss * 255
# noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
#
# # 非局部均值滤波去噪
# denoised_image_nlm = cv2.fastNlMeansDenoisingColored(noisy_image, None, 10, 10, 7, 21)
#
# # 显示结果
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.title('Original Image')
# plt.imshow(image)
#
# plt.subplot(1, 3, 2)
# plt.title('Noisy Image')
# plt.imshow(noisy_image)
#
# plt.subplot(1, 3, 3)
# plt.title('Denoised Image - NLM')
# plt.imshow(denoised_image_nlm)
#
# plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def adjust_contrast_brightness(image, contrast, brightness):
#     """
#     调整图像的对比度和亮度。
#
#     参数:
#     - image: 输入图像
#     - contrast: 对比度系数（1.0表示不变）
#     - brightness: 亮度偏移量（0表示不变）
#
#     返回:
#     - 调整后的图像
#     """
#     # 转换图像数据类型为float32以进行精确计算
#     image = np.asarray(image, dtype=np.float32)
#
#     # 应用对比度和亮度调整
#     adjusted_image = image * contrast + brightness
#
#     # 将值截取在0到255之间，并转换回uint8类型
#     adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
#
#     return adjusted_image
#
#
# # 读取图像
# image = cv2.imread('D:/EMA-yolov8-SPPLEAN/IMG20240416165432.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # 调整对比度和亮度（示例参数：对比度1.2，亮度50）
# contrast = 1
# brightness = 0
# adjusted_image = adjust_contrast_brightness(image, contrast, brightness)
#
# # 显示原图和调整后的图像
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image)
#
# plt.subplot(1, 2, 2)
# plt.title('Adjusted Image')
# plt.imshow(adjusted_image)
#
# plt.show()
#
# # 保存调整后的图像
# adjusted_image_bgr = cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2BGR)
# cv2.imwrite('adjusted_image.jpg', adjusted_image_bgr)


# import cv2
# import numpy as np
#
# # 读取彩色图像
# image = cv2.imread('D:/EMA-yolov8-SPPLEAN/IMG20240416165432.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # 对比度增强（使用直方图均衡化）
# image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
# image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
# enhanced_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
#
# # 保存对比度增强后的图像
# cv2.imwrite('enhanced_image.jpg', cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
#
# # 双边滤波处理
# bilateral_filtered_image = cv2.bilateralFilter(enhanced_image, d=9, sigmaColor=75, sigmaSpace=75)
#
# # 保存双边滤波处理后的图像
# cv2.imwrite('bilateral_filtered_image.jpg', cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_RGB2BGR))
#
# # 显示图像
# cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# cv2.imshow('Enhanced Image', cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
# cv2.imshow('Bilateral Filtered Image', cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
image = cv2.imread('D:/EMA-yolov8-SPPLEAN/IMG20240416165432dbd100.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # 添加噪声
# row, col, ch = image.shape
# mean = 0
# var = 0.01
# sigma = var ** 0.5
# gauss = np.random.normal(mean, sigma, (row, col, ch))
# noisy_image = image + gauss * 255
# noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# 双边滤波去噪
denoised_image_bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# 保存
cv2.imwrite('bilateral_image.jpg', cv2.cvtColor(denoised_image_bilateral, cv2.COLOR_RGB2BGR))

# # 显示结果
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.title('Original Image')
# plt.imshow(image)
#
# plt.subplot(1, 3, 2)
# plt.title('Noisy Image')
# plt.imshow(noisy_image)
#
# plt.subplot(1, 3, 3)
# plt.title('Denoised Image - Bilateral')
# plt.imshow(denoised_image_bilateral)
#
# plt.show()
