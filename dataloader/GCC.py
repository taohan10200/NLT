##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/Sha-Lab/FEAT
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from dataloader.setting import cfg_data
import pandas as pd
from glob import  glob


class GCC(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname,  main_transform=None,img_transform=None,gt_transform=None,filter_rule=None,IFS_path=None):
        # Set the path according to train, val and test
        if setname=='train':
            txt_path = osp.join(cfg_data.GCC_scene_dir, 'all')
            scenes_list = os.listdir(txt_path)
        elif setname=='test':
            txt_path = osp.join(cfg_data.GCC_scene_dir, 'test')
            scenes_list = os.listdir(txt_path)
        elif setname=='val':
            txt_path = osp.join(cfg_data.GCC_scene_dir, 'val')
            scenes_list = os.listdir(txt_path)
        elif '.txt' in setname:
            txt_path = osp.join(cfg_data.GCC_scene_dir, 'val')
            scenes_list = [setname]
        else:
            raise ValueError('Wrong setname.')

        data = []
        label = []
        self.IFS_path = IFS_path

        self.crowd_level = []  # 拥挤程度
        self.time = []  # 拍摄时间
        self.weather = []  # 天气状况
        self.file_folder = []  # 所属文件夹
        self.file_name = []  # 文件名字
        self.gt_cnt = []  # 标记人头数

        # self.mode = mode
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        self.max_list = cfg_data.MAX_LIST
        # Get folders' name
        txt_list = [osp.join(txt_path, scene_name) for scene_name in scenes_list if osp.isfile(osp.join(txt_path, scene_name))]

        # Get the images' paths and labels
        label_idx = 0
        for idx, this_list in enumerate(txt_list):
            with open(this_list) as f:
                lines = f.readlines()  # Read all lines available on the input stream and return them as a list.
            # print(lines)
            sub_count = 0
            filter_lines = []
            for line in lines:
                splited = line.strip().split()
                if filter_rule is not None:
                    if not self.get_filter_flag(splited, filter_rule):
                        continue
                sub_count += 1
                filter_lines.append(splited)

            if sub_count>2:
                for line in filter_lines:
                    splited = line
                    self.crowd_level.append(splited[0])
                    self.time.append(splited[1])
                    self.weather.append(splited[2])
                    self.file_folder.append(splited[3])
                    self.file_name.append(splited[4])
                    self.gt_cnt.append(int(splited[5]))
                    label.append(label_idx)
                label_idx  +=1
                # print(label_idx, sub_count)
        self.num_samples = len(self.file_name)
        print(setname,self.num_samples)


        # Set data, label and class number to be accessable from outside
        self.data = self.file_name
        self.label = label
        self.num_scenes = len(set(label))
        print(self.num_scenes)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # path, label = self.data[i], self.label[i]
        img, den = self.read_image_and_gt(index)

        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)

            # den = torch.from_numpy(np.array(den, dtype=np.float32))
        if self.gt_transform is not None:
            den = self.gt_transform(den)

        img_path = os.path.join(self.file_folder[index],'pngs_544_960',self.file_name[index] + '.png')
        return img, den

    def read_image_and_gt(self, index):
        # print self.file_folder[index] + ' ' + self.file_name[index] + '
        if self.IFS_path is not None:
            img_path = os.path.join(self.IFS_path,self.file_name[index]+'s2t.png')
        else:
            img_path = os.path.join(cfg_data.DATA_PATH+self.file_folder[index], 'pngs_544_960', self.file_name[index]+'.png')
        # ','equals '/'
        den_map_path = os.path.join(cfg_data.DATA_PATH + self.file_folder[index], 'csv_den_maps_' + cfg_data.DATA_GT + '_544_960',self.file_name[index]+ '.csv')

        img = Image.open(img_path)
        den_map = pd.read_csv(den_map_path, sep=',', header=None).values
        den_map = den_map.astype(np.float32, copy=False)

       # print(img.mode) # out: RGB
        den_map = Image.fromarray(den_map)

        # print(img_path)

        return img, den_map

    def get_num_samples(self):
        return self.num_samples

    def get_filter_flag(self, info, filter_rule):
        # pdb.set_trace()
        if not (int(info[0]) in filter_rule["level"]):
            return False
        if not (
            int(info[1]) >= filter_rule["time_duration"][0]
            and int(info[1]) <= filter_rule["time_duration"][1]
        ):
            return False
        if not (int(info[2]) in filter_rule["weather"]):
            return False
        if not (
            int(info[5]) >= filter_rule["cnt_range"][0]
            and int(info[5]) <= filter_rule["cnt_range"][1]
        ):
            return False

        cur_ratio = float(info[5]) / self.max_list[int(info[0])]
        if not (cur_ratio > filter_rule["min_ratio"]):
            return False

        return True
