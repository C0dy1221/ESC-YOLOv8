# -*- coding: utf-8 -*-
import json
import os
import argparse
from tqdm import tqdm


def convert_label_json(json_dir, save_dir, classes):
    json_paths = os.listdir(json_dir)
    classes = classes.split(',')

    for json_path in tqdm(json_paths):
        # linedata = ""
        # for json_path in json_paths:
        path = os.path.join(json_dir, json_path)
        print(path)

        with open(path, "r", encoding="utf-8") as load_f:
            json_dict = json.load(load_f)
        h, w = json_dict['imageHeight'], json_dict['imageWidth']

        txt_path = os.path.join(save_dir, json_path.replace('json', 'txt'))
        txt_file = open(txt_path, 'a+')
        for shape_dict in json_dict['shapes']:
            print("shape_dict['label']:"+shape_dict['label'])
            label = shape_dict['label']
            if label in classes:
                # label_index = classes.index(label)
                points = shape_dict['points']

                points_nor_list = []

                for point in points:
                    points_nor_list.append(round(point[0] / w,6))
                    points_nor_list.append(round(point[1] / h,6))

                points_nor_list = list(map(lambda x: str(x), points_nor_list))
                points_nor_str = ' '.join(points_nor_list)

                label_str = str(1) + ' ' + points_nor_str + '\n'
                # save txt path

                txt_file.writelines(label_str)

            else:
                with open("D:/00a/singlesickfish/sick/nodone.txt",'a+') as c:
                    c.writelines(path+"\n")
        txt_file.close()


if __name__ == "__main__":
    """
    python json2txt.py --json-dir my_datasets/color_rings/jsons --save-dir my_datasets/color_rings/txts --classes "cat,dogs"
    """
    parser = argparse.ArgumentParser(description='json convert to txt params')
    parser.add_argument('--json-dir', type=str, default='D:/00a/singlesickfish/sick/json', help='json path dir')
    parser.add_argument('--save-dir', type=str, default='D:/00a/singlesickfish/sick/txt', help='txt save dir')
    # parser.add_argument('--classes', type=str, default='OZ,oz,cz,CZ,zebrafish, tm', help='classes')
    parser.add_argument('--classes', type=str, default='zf,OZ,oz,cz,CZ,zebrafish,back, tm,szf,tail', help='classes')
    args = parser.parse_args()
    json_dir = args.json_dir
    save_dir = args.save_dir
    classes = args.classes
    convert_label_json(json_dir, save_dir, classes)