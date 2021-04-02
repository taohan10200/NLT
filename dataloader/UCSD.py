import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
import pandas as pd
from dataloader.setting import  cfg_data
class UCSD(data.Dataset):
    def __init__(self, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.mode =mode
        # self.data_files = [filename for filename in os.listdir(self.img_path) \
        #                    if os.path.isfile(os.path.join(self.img_path,filename))]
        with open(os.path.join(cfg_data.UCSD_scene_dir,mode,mode+'.txt')) as f:
            lines = f.readlines()

        self.data_files = []
        for line in lines:
            line = line.strip('\n')
            self.data_files.append(line)

        if self.mode == 'train':
            self.data_files = int(cfg_data.num_batch * cfg_data.target_shot_size / (len(self.data_files))) * self.data_files

        self.num_samples = len(self.data_files)
        if self.mode is 'train':
            print('[UCSD DATASET]: %d training images.' % (self.num_samples))
        if self.mode is 'val':
            print('[UCSD DATASET]: %d validation images.' % (self.num_samples))
        if self.mode is 'test':
            print('[UCSD DATASET]: %d testingn images.' % (self.num_samples))

        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)

        if self.main_transform is not None:
            img, den = self.main_transform(img,den)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.gt_transform is not None:
            den = self.gt_transform(den)

        return img,  den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        if self.mode == 'train' or  'val':
            img_path = cfg_data.UCSD_DATA_PATH + '/train/img/' + fname +'.png'
            den_path = cfg_data.UCSD_DATA_PATH + '/train/den/' + fname + '.csv'
        if self.mode == 'test':
            img_path = cfg_data.UCSD_DATA_PATH + '/test/img/' +  fname
            den_path = cfg_data.UCSD_DATA_PATH + '/test/den/' + os.path.splitext(fname)[0] + '.csv'
        img = Image.open(img_path)

        if img.mode == 'L':
            img = img.convert('RGB')

        # den = sio.loadmat(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        den = pd.read_csv(den_path, sep=',',header=None).values
        
        den = den.astype(np.float32, copy=False)    
        den = Image.fromarray(den)  
        return img, den    

    def get_num_samples(self):
        return self.num_samples       
            
        