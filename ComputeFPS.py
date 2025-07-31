# import time
# import cv2
# import torch
# from ultralytics import YOLO
#
# # 加载YOLOv8模型
# model = YOLO('D:/EMA-yolov8-SPPLEAN/runs/segment/train7/weights/best.pt')  # 使用YOLOv8n模型，你也可以选择其他权重文件
#
# # 打开摄像头或视频文件
# # cap = cv2.VideoCapture(0)  # 使用摄像头
# cap = cv2.VideoCapture('path_to_your_video_file.mp4')  # 使用视频文件
#
# # 初始化计数器
# frame_count = 0
# total_inference_time = 0
#
# # 读取视频帧并进行推理
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     start_time = time.time()
#
#     # 推理
#     results = model(frame)
#
#     end_time = time.time()
#
#     # 计算每帧处理时间
#     inference_time = end_time - start_time
#     total_inference_time += inference_time
#     frame_count += 1
#
#     # 获取检测结果
#     annotated_frame = results.render()  # 渲染结果
#
#     # 显示带注释的帧
#     cv2.imshow('YOLOv8 Detection', annotated_frame[0])
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 计算平均FPS
# average_fps = frame_count / total_inference_time
# print(f'Average FPS: {average_fps:.2f}')
#
# # 释放资源
# cap.release()
# cv2.destroyAllWindows()


import time
import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('D:/EMA-yolov8-SPPLEAN/runs/segment/train7/weights/best.pt')  # 使用YOLOv8n模型，你也可以选择其他权重文件

# 读取测试图片
image_path = 'D:/EMA-yolov8-SPPLEAN/add_3d_short2_2476.jpg'
image = cv2.imread(image_path)

# 推理次数
num_inferences = 100

# 初始化计时器
total_inference_time = 0

# 进行多次推理以计算平均推理时间
for _ in range(num_inferences):
    start_time = time.time()

    # 推理
    results = model(image)

    end_time = time.time()

    # 计算每次推理的时间
    inference_time = end_time - start_time
    total_inference_time += inference_time

# 计算平均FPS
average_fps = num_inferences / total_inference_time
print(f'Average FPS: {average_fps:.2f}')

# # 获取检测结果
# annotated_image = results.render()
#
# # 显示带注释的图片
# cv2.imshow('YOLOv8 Detection', annotated_image[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
