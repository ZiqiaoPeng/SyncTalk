# ref:https://github.com/ShunyuYao/DFA-NeRF
import sys
import os
from tqdm import tqdm
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'core'))
from pathlib import Path
from data_test_flow import *
from models.network_test_flow import NeuralNRT
from options_test_flow import TestOptions
import torch
import numpy as np



def save_flow_numpy(filename, flow_input):
    np.save(filename, flow_input)


def predict(data):
    with torch.no_grad():
        model.eval()
        path_flow = data["path_flow"]
        src_crop_im = data["src_crop_color"].cuda()
        tar_crop_im = data["tar_crop_color"].cuda()
        src_im = data["src_color"].cuda()
        tar_im = data["tar_color"].cuda()
        src_mask = data["src_mask"].cuda()
        crop_param = data["Crop_param"].cuda()
        B = src_mask.shape[0]
        flow = model(src_crop_im, tar_crop_im, src_im, tar_im, crop_param)
        for i in range(B):
            flow_tmp = flow[i].cpu().numpy() * src_mask[i].cpu().numpy()
            save_flow_numpy(os.path.join(save_path, os.path.basename(
                path_flow[i])[:-6]+".npy"), flow_tmp)


if __name__ == "__main__":
    width = 272
    height = 480

    test_opts = TestOptions().parse()
    test_opts.pretrain_model_path = os.path.join(
        dir_path, 'pretrain_model/raft-small.pth')
    data_loader = CreateDataLoader(test_opts)
    testloader = data_loader.load_data()
    model_path = os.path.join(dir_path, 'sgd_NNRT_model_epoch19008_50000.pth')
    model = NeuralNRT(test_opts, os.path.join(
        dir_path, 'pretrain_model/raft-small.pth'))
    state_dict = torch.load(model_path)

    model.CorresPred.load_state_dict(state_dict["net_C"])
    model.ImportanceW.load_state_dict(state_dict["net_W"])

    model = model.cuda()

    save_path = test_opts.savepath
    Path(save_path).mkdir(parents=True, exist_ok=True)
    total_length = len(testloader)

    for batch_idx, data in tqdm(enumerate(testloader), total=total_length):
        predict(data)
