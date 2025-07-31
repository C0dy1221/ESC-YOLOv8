# # 从每个文件中获取第一张图像
# import os
# import shutil
#
# # 源文件夹和目标文件夹路径
# source_folder = "D:/arcface-pytorch-main/datasets/back50test/"
# destination_folder = "C:/Users/DELL/Desktop/picjoin_madepic/"
#
# # 遍历源文件夹中的每个子文件夹
# for folder_name in os.listdir(source_folder):
#     folder_path = os.path.join(source_folder, folder_name)
#
#     # 检查子文件夹是否是目标文件夹
#     if not os.path.isdir(folder_path):
#         continue
#
#     # 获取子文件夹中的第一张图像文件路径
#     image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
#     if len(image_files) < 1:
#         continue
#
#     first_image_path = os.path.join(folder_path, image_files[0])
#
#     # 创建目标文件夹（如果不存在）
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#
#     # 复制图像文件到目标文件夹
#     destination_path = os.path.join(destination_folder, f"{folder_name}_image.jpg")
#     shutil.copyfile(first_image_path, destination_path)
#     print(f"已将 {first_image_path} 复制到 {destination_path}")
#
# print("所有图像复制完成")


# 下面这个代码用来拼接所有的单个图像
from PIL import Image
import os

# 输入文件夹路径
input_folder = "C:/Users/DELL/Desktop/picjoin_madepic/"

# 获取文件夹中所有图片文件的路径
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 计算所需的行数和列数
num_rows = 14
num_columns = 10

# 创建一个列表来保存所有图像对象
images = []

# 逐个打开图片文件并添加到列表中
flag = 0   # 用来获取第一张图像的大小
for image_file in image_files:
    flag = flag + 1
    if flag == 1:
        img = Image.open(image_file)
        width, height = img.size
    img = Image.open(image_file)
    img_resize = img.resize((width,height))
    images.append(img_resize)

# # 获取第一张图片的大小
# first_image = images[0]
# width, height = first_image.size

# 计算新图像的大小
new_width = num_columns * width
new_height = num_rows * height

# 创建一个新的大图像对象
new_image = Image.new('RGB', (new_width, new_height))

# 将每张图片依次粘贴到大图像上
for i, img in enumerate(images):
    row = i // num_columns
    col = i % num_columns
    x_offset = col * width
    y_offset = row * height
    new_image.paste(img, (x_offset, y_offset))

# 保存拼接好的图片
output_path = "C:/Users/DELL/Desktop/output/image_concatenated50.jpg"
new_image.save(output_path)

print(f"拼接完成，结果保存在 {output_path}")


