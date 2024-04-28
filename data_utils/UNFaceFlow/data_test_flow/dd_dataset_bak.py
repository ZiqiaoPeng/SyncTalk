import os.path
import torch
import torch.utils.data as data
from PIL import Image
import random
import utils
import numpy as np
import torchvision.transforms as transforms
from utils_core import flow_viz
import cv2

class DDDataset(data.Dataset):
    def __init__(self):
        super(DDDataset, self).__init__()
    def initialize(self, opt):
        self.opt = opt
        self.dir_txt = opt.datapath
        self.paths = []
        in_file = open(self.dir_txt, "r")
        k = 0
        list_paths = in_file.readlines()
        for line in list_paths:
            #if k>=20: break
            flag = False
            line = line.strip()
            line = line.split()
            
            #source data
            if (not os.path.exists(line[0])):
                print(line[0]+" not exists")
                continue
            if (not os.path.exists(line[1])):
                print(line[1]+" not exists")
                continue
            if (not os.path.exists(line[2])):
                print(line[2]+" not exists")
                continue
            if (not os.path.exists(line[3])):
                print(line[3]+" not exists")
                continue
            # if (not os.path.exists(line[2])):
            #     print(line[2]+" not exists")
            #     continue

            # path_list = [line[0], line[1], line[2]]
            path_list = [line[0], line[1], line[2], line[3]]
            self.paths.append(path_list)
            k += 1
        in_file.close()
        self.data_size = len(self.paths)
        print("num data: ", len(self.paths))

    def process_data(self, color, mask):
        non_zero = mask.nonzero()
        bound = 10
        min_x = max(0, non_zero[1].min()-bound)
        max_x = min(self.opt.width-1, non_zero[1].max()+bound)
        min_y = max(0, non_zero[0].min()-bound)
        max_y = min(self.opt.height-1, non_zero[0].max()+bound)
        color = color * (mask!=0).astype(float)[:, :, None]
        crop_color = color[min_y:max_y, min_x:max_x, :]
        crop_color = cv2.resize(np.ascontiguousarray(crop_color), (self.opt.crop_width, self.opt.crop_height), interpolation=cv2.INTER_LINEAR)
        crop_params = [[min_x], [max_x], [min_y], [max_y]]

        return crop_color, crop_params

    def __getitem__(self, index):
        paths = self.paths[index % self.data_size]
        src_color = np.array(Image.open(paths[0]))
        src_color = src_color.astype(np.uint8)
        raw_src_color = src_color.copy()
        src_mask = np.array(Image.open(paths[1]))
        src_mask_copy = src_mask.copy()
        src_crop_color, src_crop_params = self.process_data(src_color, src_mask)
        #self.write_mesh(src_X, src_Y, src_Z, "./tmp/src.obj")
        #HWC --> CHW, 
        raw_src_color = torch.from_numpy(raw_src_color).permute(2, 0, 1).float() / 255.0
        src_crop_color = torch.from_numpy(src_crop_color).permute(2, 0, 1).float() / 255.0

        src_mask_copy = (src_mask_copy!=0)
        src_mask_copy = torch.tensor(src_mask_copy[np.newaxis, :, :])

        tar_color = np.array(Image.open(paths[2]))
        tar_color = tar_color.astype(np.uint8)
        raw_tar_color = tar_color.copy()
        tar_mask = np.array(Image.open(paths[3]))
        tar_mask_copy = tar_mask.copy()
        tar_crop_color, tar_crop_params = self.process_data(tar_color, tar_mask) 

        raw_tar_color = torch.from_numpy(raw_tar_color).permute(2, 0, 1).float() / 255.0
        tar_crop_color = torch.from_numpy(tar_crop_color).permute(2, 0, 1).float() / 255.0

        tar_mask_copy = (tar_mask_copy!=0)
        tar_mask_copy = torch.tensor(tar_mask_copy[np.newaxis, :, :])

        Crop_param = torch.tensor(src_crop_params+tar_crop_params)

        split_ = paths[0].split("/")
        path1 = split_[-1][:-4] + "_" + paths[2].split("/")[-1][:-4] +".oflow"

        return {"path_flow":path1, "src_crop_color":src_crop_color, "tar_crop_color":tar_crop_color, "src_color":raw_src_color, "tar_color":raw_tar_color, "src_mask":src_mask_copy, "tar_mask":tar_mask_copy, "Crop_param":Crop_param}

    def __len__(self):
        return self.data_size

    def name(self):
        return 'DDDataset'