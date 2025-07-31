import cv2
import os
def save_img():
  dir_path = 'F:/G/sickfish/'
  # video_path = 'D:/001zfvideo/0324/'E:\fishwatcher\0327\E2
  # video_path = 'E:/fishwatcher/0518/lalalal/'
  dirs = os.listdir(dir_path)
  for dir in dirs:
    video_path = dir_path + dir
    print(video_path)
    videos = os.listdir(video_path)
    print(videos)
    for video_name in videos:
      file_name = video_name.split('.')[0]
      folder_name = video_path +'/'+ file_name
      os.makedirs(folder_name,exist_ok=True)
      vc = cv2.VideoCapture(video_path+'/'+video_name) #读入视频文件
      c=0
      rval=vc.isOpened()
      while rval:  #循环读取视频帧
        c = c + 1
        rval, frame = vc.read()
        pic_path = folder_name+'/'
        if rval:
          cv2.imwrite(pic_path + file_name + '_' + str(c) + '.jpg', frame) #存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
          cv2.waitKey(1)
        else:
          break
      vc.release()
      print('save_success')
      print(folder_name)
save_img()
