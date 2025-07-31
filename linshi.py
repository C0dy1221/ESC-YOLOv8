# import os
# path = 'D:/yolov8datasets/grasscarp/labels/val/'
# txtfiles = os.listdir(path)
# for txtfile in txtfiles:
#     txtpath = path+txtfile
#     allinfo =[]
#     with open(txtpath,'r') as f1:
#         line = f1.read().splitlines()   # 读取文本文件，去掉后年的换行符，一般是一行
#         print(line)
#     for i in line:
#         allinfo.append(i.split(' '))
#     allinfo = allinfo[0]
#     while '' in allinfo:
#         allinfo.remove('')
#     print(allinfo)
#     allinfo = list(map(lambda x: str(round(float(x), 6)), allinfo))
#     allinfo[0] = '0'
#     allinfo.insert(7,format(2.000000, '.6f'))
#     allinfo.insert(10, format(2.000000, '.6f'))
#     allinfo.insert(13, format(2.000000, '.6f'))
#     allinfo.insert(16, format(2.000000, '.6f'))
#     allinfo.insert(19, format(2.000000, '.6f'))
#     allinfo.insert(22, format(2.000000, '.6f'))
#     allinfo.insert(25, format(2.000000, '.6f'))
#     allinfo.insert(28, format(2.000000, '.6f'))
#     allinfo.insert(31, format(2.000000, '.6f'))
#     allinfo = ' '.join(allinfo)
#     print(allinfo)
#     with open(txtpath,'w') as f:
#         f.write(allinfo+'\n')
#
#

# 改名
# import os
# path0 = 'D:/01grasscarpleftandright/0530/'
# dirlist = os.listdir(path0)
# for dir in dirlist:      # dir是文件夹命名A几B几
#     path = path0+dir+'/'
#     delldir = os.listdir(path0+dir)
#     for dellfile in delldir:
#         if dellfile == 'saveright':
#             # 图片路径
#             imagepath = path+dellfile
#             imagelist = os.listdir(imagepath)
#             for image in imagelist:
#                 prefix = dir+'_0530_right_'
#                 oldname = imagepath + '/' + image
#                 newname = imagepath + '/' + prefix+image
#                 os.rename(oldname,newname)

# 改名
# import os
# path0 = 'D:/yolov8datasets/1/back50test2/'
# dirlist = os.listdir(path0)   # 日期 0504 0525 ……
# for dir in dirlist:
#     path = path0+dir
#     datalist = os.listdir(path)    # 文件名 E1 E2 ……
#     for datadir in datalist:
#         path1 = path+'/'+datadir
#         imagefiledir = os.listdir(path1)
#         for imagefile in imagefiledir:
#             if 'back' in imagefile:
#                 path3 = path1+'/' + imagefile
#                 imagelist = os.listdir(path3)
#                 for image in imagelist:
#                     prefix = datadir + '_' +dir + '_'
#                     oldname = path3 + '/' + image
#                     print(oldname)
#                     newname = path3 + '/' + prefix + image
#                     print(newname)
#                     os.rename(oldname,newname)

# 移动文件
# import os
# import shutil
# path0 = 'D:/yolov8datasets/1/back40test2/'
# dirlist = os.listdir(path0)   # 日期 0504 0525 ……
# for dir in dirlist:
#     path = path0+dir
#     datalist = os.listdir(path)    # 文件名 E1 E2 ……
#     for datadir in datalist:
#         path1 = path+'/'+datadir
#         imagefiledir = os.listdir(path1)
#         for imagefile in imagefiledir:
#             if 'back' in imagefile:
#                 path3 = path1+'/' + imagefile
#                 imagelist = os.listdir(path3)
#                 for image in imagelist:
#                     print()
#                     imagepath = path3+ '/' +image
#                     movepath = path1
#                     print(imagepath)
#                     print(movepath)
#                     shutil.move(imagepath,movepath)
# import os
# import shutil
# path0 = 'D:/yolov8datasets/1/back40test2/'
# dirlist = os.listdir(path0)   # 日期 0504 0525 ……
# for dir in dirlist:
#     path = path0+dir
#     datalist = os.listdir(path)    # 文件名 E1 E2 ……
#     for datadir in datalist:
#         path1 = path+'/'+datadir
#         imagefiledir = os.listdir(path1)
#         for imagefile in imagefiledir:
#             if '.jpg' in imagefile:
#                 movepath = path1+'/'+'back'
#                 imagepath = path1 + '/' + imagefile
#                 print(movepath)
#                 print(imagepath)
#                 shutil.move(imagepath,movepath)
# 移动
# import os
# import shutil
#
# path0 =  'D:/yolov8datasets/1/back50test2/'
# dirlist = os.listdir(path0)   #  文件名 E1 E2 ……
# for dir in dirlist:
#     path1 = path0+dir   # 移到这里文件
#     segdir = os.listdir(path1)
#     for segfile in segdir:
#         if 'segdata' in segfile:
#             path2 = path1 + '/' +segfile
#             imagelist = os.listdir(path2)
#             for image in imagelist:
#                 imagepath = path2 + '/' + image
#                 print(path1)
#                 print(imagepath)
#                 shutil.move(imagepath,path1)


# 移动图片
# import os
# import shutil
#
# # 原始文件夹路径
# original_folder = "D:/yolov8datasets/1/back50test2/0405/"
#
#
# # 遍历原始文件夹中的文件
# for filename in os.listdir(original_folder):    # 这下面是E1，E2……
#     # 新建文件夹路径
#     new_folder = os.path.join(original_folder,filename+"/"+"back")
#     print(new_folder)
#     # 如果新建文件夹不存在，则创建
#     if not os.path.exists(new_folder):
#         os.makedirs(new_folder)
#     imagespath = os.path.join(original_folder,filename)
#     for image in os.listdir(imagespath):
#         # 检查文件是否为图像文件（这里假设图像文件的扩展名为.jpg）
#         if image.endswith(".jpg"):
#             # 构建原始文件路径和目标文件路径
#             src_path = os.path.join(imagespath, image)
#             dest_path = os.path.join(new_folder, image)
#             # 移动图像文件到新建文件夹中
#             shutil.move(src_path, dest_path)
#
# print("图像已成功移动到新建文件夹中")


# 移动到文件夹
# import os
# import shutil
# path0 = 'D:/yolov8datasets/0/'
# #path0 = "D:/arcface-pytorch-main/datasets/singlebackdata/70days/"
# p = 'D:/yolov8datasets/back76test/'
# dirlist = os.listdir(path0)   #  文件名 E1_*.jpg E2_*.jpg ……
# for dir in dirlist:
#     imagepath = path0+dir
#     path1 = p+dir.split('_',1)[0]+'/'  # 移到这里文件
#     print(imagepath)
#     print(path1)
#     shutil.move(imagepath,path1)

# 把文件夹里的图像合并成视频
# import cv2
# import os
# image_folder = 'D:/00a/0/11-2/'   # 图像文件件路径
# video_path = 'D:/00a/0/9.mp4'     # 输出视频文件路径
# fps = 30    # 视频帧率
#
# images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# frame = cv2.imread(os.path.join(image_folder,images[0]))
# height, width, layers = frame.shape
#
# video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#
# cv2.destroyAllWindows()
# video.release()



# 这个是把鱼背图片的存储空间减小
# from PIL import Image
# import os
# path = 'D:/BaiduNetdiskDownload/new--ok/pic/'
# savepath = 'D:/BaiduNetdiskDownload/new--ok/lpic/'
# piclist = os.listdir(path)
# for pic in piclist:
#     picpath = path + pic
#     savepic = savepath + pic
#     I = Image.open(picpath)
#     I.save(savepic)

# import json
# from PIL import Image
# path0 = 'D:/00a/1test/'
# pic = "A1_0509_000.jpg"
# picpath = path0 + pic
# # 获取图像的大小
# image = Image.open(picpath)
# image_width, image_height = image.size
# # 示例的YOLOv8模型输出
# yolov8_output = [
#     {"class": "person", "bbox": [100, 100, 200, 200]},
#     {"class": "car", "bbox": [300, 300, 400, 400]}
# ]

# # 构建Labelme标注的JSON格式
# labelme_json = {
#     "version": "4.5.6",
#     "flags": {},
#     "shapes": [],
#     "imagePath": pic,
#     "imageData": None,
#     "imageHeight": image_height,
#     "imageWidth": image_width
# }
#
# # 将YOLOv8输出转换为Labelme标注格式
# for detection in yolov8_output:
#     labelme_json["shapes"].append({
#         "label": detection["class"],
#         "points": [[detection["bbox"][0], detection["bbox"][1]], [detection["bbox"][2], detection["bbox"][3]]],
#         "shape_type": "rectangle",
#         "flags": {}
#     })
#
# # 将Labelme标注输出写入文件
# with open("labelme_annotation.json", "w") as json_file:
#     json.dump(labelme_json, json_file, indent=4)


# import os
# def check_single_line_text_files(folder_path):
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         # 判断文件是否是普通文件并且是文本文件
#         if os.path.isfile(file_path) and filename.endswith('.txt'):
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()
#                 # 检查文件是否只有一行内容
#                 if len(lines) == 1:
#                     continue
#                 else:
#                     print(f"File '{filename}' does not have only one line.")
# # 检查指定文件夹中的文本文件是否只有一行内容
# check_single_line_text_files('D:/yolov8datasets/sickfishORC/afteryhcheck/txt')


# import os
#
# # 地址
# path = "D:/arcface-pytorch-main/datasets/102/K8"
#
# # 提取最后一层文件名
# filename = os.path.basename(path)
# print(os.path.dirname(path))
# print("最后一层文件名：", filename)

# 假设你有一个字典列表，键不同，但值都是列表
# dict_list = [
#     {'key1': [3, 2, 1]},
#     {'key2': [1, 4, 2]},
#     {'key3': [2, 1, 3]}
# ]

# 定义一个排序函数，用于按照值列表的第一个值进行排序
# def sort_by_first_value(d):
#     key = list(d.keys())[0]  # 获取字典的键
#     return d[key][0]
#
# # 对字典列表进行排序
# sorted_dict_list = sorted(dict_list, key=sort_by_first_value)
#
# print("按照值列表的第一个值排序后的字典列表：", sorted_dict_list)

# 删除json中的imageData内容
# import json
# import os
# path = 'D:/00a/last/sick/1/'
# jsonlist = [json for json in os.listdir(path) if json.endswith('json')]
# print(jsonlist)
# for jsonfile in jsonlist:
#     jsonpath = path + jsonfile
#     # 读取JSON文件
#     with open(jsonpath, 'r') as f:
#         data = json.load(f)
#     # 删除imageData字段
#     if 'imageData' in data:
#         data['imageData'] = None
#     # 将修改后的数据保存为新的JSON文件
#     with open(jsonpath, 'w') as f:
#         json.dump(data, f, indent=4)  # 使用indent参数将数据格式化为可读的JSON格式

# 修改json文件中label名字
# import json
# import os
# path = 'D:/00a/last/sick/images/'
# jsonlist = [json for json in os.listdir(path) if json.endswith('json')]
# print(jsonlist)
# for jsonfile in jsonlist:
#     jsonpath = path + jsonfile
#     with open(jsonpath, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         # 遍历shapes并修改标签
#         for shape in data['shapes']:
#             if shape['label'] == 'szf':
#                 shape['label'] = 'sick_zf'
#             else:
#                 shape['label'] = 'zf'
#
#         # 保存修改后的JSON文件
#         with open(jsonpath, 'w') as f:
#             json.dump(data, f,indent=4)

# 修改json文件中imagePath的路径
# import json
# import os
# path = 'D:/00a/last/sick/1/'
# jsonlist = [json for json in os.listdir(path) if json.endswith('json')]
# print(jsonlist)
# for jsonfile in jsonlist:
#     jsonpath = path + jsonfile
#     # 读取JSON文件
#     with open(jsonpath, 'r') as f:
#         data = json.load(f)
#     # 修改imagePath
#     if 'imagePath' in data:
#         data['imagePath'] = jsonfile.split('.')[0]+'.jpg'
#     # 将修改后的数据保存为新的JSON文件
#     with open(jsonpath, 'w') as f:
#         json.dump(data, f, indent=4)  # 使用indent参数将数据格式化为可读的JSON格式



# 新的改名
# import os
# path0 = 'D:/00a/singlesickfish/diedfish/'
# dirlist = os.listdir(path0)   # 1,3,5,6 ……
# for dir in dirlist:
#     path = path0+dir
#     datalist = os.listdir(path)    # 图片list
#     for datadir in datalist:   # 图片循环
#         path1 = path+'/'+datadir   #图片的path+name
#         prefix = 'diedfish_'+ dir +'_'
#         oldname = path1
#         print(oldname)
#         newname = path + '/' +prefix+datadir
#         print(newname)
#         os.rename(oldname, newname)

# # 把图片的存储空间改小
# from PIL import Image
# import os
# path = 'D:/00a/singlesickfish/sick/images/'
# savepath = 'D:/00a/singlesickfish/sick/saveimages/'
# piclist = os.listdir(path)
# for pic in piclist:
#     picpath = path + pic
#     savepic = savepath + pic
#     I = Image.open(picpath)
#     I.save(savepic)

# import os
# path0 = 'D:/00a/last/sick/morefishimages/1/'
# filelist = os.listdir(path0)
# for file in filelist:
#     oldname = path0+file
#     print(oldname)
#     newname = path0+"more_"+file
#     print(newname)
#     os.rename(oldname, newname)

# import json
# import os
# path = 'D:/00a/singlesickfish/wt/json/'
# jsonlist = [json for json in os.listdir(path) if json.endswith('json')]
# print(jsonlist)
# for jsonfile in jsonlist:
#     jsonpath = path + jsonfile
#     with open(jsonpath, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         # 遍历shapes并修改标签
#         sflag = 0
#         flag = 0
#         for shape in data['shapes']:
#             if shape['label'] == 'sick_zf':
#                 sflag = sflag + 1
#                 shape['label'] = 'sickzf'+'_'+str(sflag)
#             else:
#                 flag = flag + 1
#                 shape['label'] = 'zf'+'_'+str(flag)
#
#
#         # 保存修改后的JSON文件
#         with open(jsonpath, 'w') as f:
#             json.dump(data, f,indent=4)


# import matplotlib.pyplot as plt
#
# bwith =2
# # 图一数据
# models1 = ['YOLOv8-seg', 'YOLOv8seg + CA', 'YOLOv8seg + EMA', 'YOLOv8seg + SPPELAN', 'ESC-YOLOv8-seg']
# y1 = [0.941, 0.941, 0.955, 0.960, 0.969]
#
# models2 = ['YOLOv8-seg', 'YOLOv8seg + EMA', 'ESC-YOLOv8-seg']
# y2 = [0.945, 0.959, 0.977]
#
# models3 = ['YOLOv8-seg', 'YOLOv8seg + EMA', 'ESC-YOLOv8-seg']
# y3 = [0.975, 0.979, 0.984]
#
# # 图2数据
# models4 = ['YOLOv8-seg', 'YOLOv8seg + CA', 'YOLOv8seg + EMA', 'YOLOv8seg + SPPELAN', 'ESC-YOLOv8-seg']
# y4 = [0.954, 0.960, 0.960, 0.969, 0.973]
#
# models5 = ['YOLOv8-seg', 'YOLOv8seg + EMA', 'ESC-YOLOv8-seg']
# y5 = [0.956, 0.960, 0.978]
#
# models6 = ['YOLOv8-seg', 'YOLOv8seg + EMA', 'ESC-YOLOv8-seg']
# y6 = [0.980, 0.988, 0.988]
#
# # 创建图表
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
# plt.subplots_adjust(left=0.05, top= 0.90, right = 0.97, bottom = 0.1, wspace = 0.15, hspace = 0.1)
# #图一
#
# # 第一条线
# ax1.plot(models1, y1, marker='o', linestyle='--', label='Raw small target images')
# for x,y in zip(models1,y1):
#     ax1.text(x,y,str(y),fontdict={'fontsize':10})
# # 第二条线
# ax1.plot(models2, y2, marker='s', linestyle='--', label='Small target image after preprocessing')
# for x,y in zip(models2,y2):
#     ax1.text(x,y,str(y),fontdict={'fontsize':10})
# # 第三条线
# ax1.plot(models3, y3, marker='^', linestyle='--', label='All images')
# for x,y in zip(models3,y3):
#     ax1.text(x,y,str(y),fontdict={'fontsize':10})
# # 设置图1标题和标签
# ax1.set_title('Precision', fontsize=14, fontweight='bold')
# ax1.set_xlabel('Models', fontsize=12)
# ax1.set_ylabel('Value', fontsize=12)
# # ax1.tick_params(labelsize=10)
#
# ax1.set_ylim(0.90, 1.00)
# ax1.legend(loc=3)
# ax1.spines['bottom'].set_linewidth(bwith)
# ax1.spines['left'].set_linewidth(bwith)
# ax1.spines['top'].set_linewidth(bwith)
# ax1.spines['right'].set_linewidth(bwith)
#
# ax1.grid(linestyle='-.')
# ax1.set_xticks(models1)
# ax1.set_xticklabels(models1, rotation=10)
#
# #图2
#
# # 第一条线
# ax2.plot(models4, y4, marker='o', linestyle='--', label='Raw small target images')
# for x,y in zip(models4,y4):
#     ax2.text(x,y,str(y),fontdict={'fontsize':10})
# # 第二条线
# ax2.plot(models5, y5, marker='s', linestyle='--', label='Small target image after preprocessing')
# for x,y in zip(models5,y5):
#     ax2.text(x,y,str(y),fontdict={'fontsize':10})
# # 第三条线
# ax2.plot(models6, y6, marker='^', linestyle='--', label='All images')
# for x,y in zip(models6,y6):
#     ax2.text(x,y,str(y),fontdict={'fontsize':10})
# # 设置图1标题和标签
# ax2.set_title('Recall', fontsize=14, fontweight='bold')
# ax2.set_xlabel('Models', fontsize=12)
# ax2.set_ylabel('Value', fontsize=12)
# ax2.set_ylim(0.90, 1.00)
# ax2.legend()
# ax2.spines['bottom'].set_linewidth(bwith)
# ax2.spines['left'].set_linewidth(bwith)
# ax2.spines['top'].set_linewidth(bwith)
# ax2.spines['right'].set_linewidth(bwith)
# ax2.grid(linestyle='-.')
# ax2.set_xticks(models4)
# ax2.set_xticklabels(models4, rotation=10)
#
#
# # 显示图表
# plt.grid(linestyle='-.')
# plt.savefig('result.png')
# plt.show()


# # 随机获取一般的问价
# import os
# import random
# import shutil
#
# def get_random_half_files(directory):
#     # 获取文件夹中的所有文件
#     all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
#
#     # 计算一半的文件数量
#     half_count = len(all_files) // 2
#
#     # 随机选择一半的文件
#     random_files = random.sample(all_files, half_count)
#
#     return random_files
#
#
# # 文件夹路径
# folder_path = 'D:/00a/sickfishdataset/images/'
# txtdir_path = 'D:/00a/sickfishdataset/txt/'
# saveimagespath = 'D:/00a/sickfishdataset/halfimages/'
# savetxtpath = "D:/00a/sickfishdataset/halftxt/"
# # 获取随机选择的一半文件
# random_half_files = get_random_half_files(folder_path)
#
# for image in random_half_files:
#     imagepath = folder_path+image
#     print(imagepath)
#     txtpath = txtdir_path+image.split(".")[0]+".txt"
#     shutil.move(imagepath,saveimagespath)
#     shutil.move(txtpath,savetxtpath)
#
#
# # print(random_half_files)

import matplotlib.pyplot as plt
import numpy as np

# 示例数据
categories = ['A', 'B', 'C', 'D']
quantities1 = [100, 200, 150, 300]  # 第一组柱状图的数据
quantities2 = [120, 180, 130, 280]  # 第二组柱状图的数据
percentages = [20, 50, 35, 70]  # 折线图的百分比

# 设置柱宽
bar_width = 0.35
index = np.arange(len(categories))  # X轴位置

# 创建图表和双轴
fig, ax1 = plt.subplots()

# 在左侧Y轴画两组柱状图
bars1 = ax1.bar(index - bar_width/2, quantities1, bar_width, color='b', alpha=0.6, label='数量1')
bars2 = ax1.bar(index + bar_width/2, quantities2, bar_width, color='g', alpha=0.6, label='数量2')
ax1.set_xlabel('Category')
ax1.set_ylabel('数量', color='b')

# 添加柱状图的图例
ax1.legend(loc='upper left')

# 创建第二个Y轴用于百分比
ax2 = ax1.twinx()
ax2.plot(categories, percentages, color='r', marker='o', linestyle='-', linewidth=2, label='百分比')
ax2.set_ylabel('百分比 (%)', color='r')

# 添加折线图的图例
ax2.legend(loc='upper right')

# 设置X轴的刻度和标签
ax1.set_xticks(index)
ax1.set_xticklabels(categories)

plt.title('数量和百分比的双轴图')
plt.show()


