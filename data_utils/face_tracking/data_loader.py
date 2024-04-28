import torch
import cv2
import numpy as np
import os


def load_dir(path, start, end):
    lmss = []
    for i in range(start, end):
        datapath = os.path.join(path, str(i) + '.lms')
        if os.path.isfile(datapath):
            lms = np.loadtxt(os.path.join(
                path, str(i) + '.lms'), dtype=np.float32)
            lmss.append(lms)
        #
        # datapath = os.path.join(path, '{:d}.lms'.format(i))
        # if os.path.isfile(datapath):
        #     lms = np.loadtxt(os.path.join(
        #         path, '{:d}.lms'.format(i)), dtype=np.float32)
        #     lmss.append(lms)
    lmss = np.stack(lmss)
    lmss = torch.as_tensor(lmss).cuda()
    return lmss
