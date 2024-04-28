# ref: https://github.com/ShunyuYao/DFA-NeRF
from numpy.core.numeric import require
from numpy.lib.function_base import quantile
import torch
import numpy as np
from facemodel import Face_3DMM
from data_loader import load_dir
from util import *
import os
import sys
import cv2
import imageio
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))


def set_requires_grad(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = True


parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, default="obama/ori_imgs", help="idname of target person")
parser.add_argument('--img_h', type=int, default=512, help='image height')
parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--frame_num', type=int,
                    default=11000, help='image number')
args = parser.parse_args()
start_id = 0
end_id = args.frame_num

lms = load_dir(args.path, start_id, end_id)
num_frames = lms.shape[0]
h, w = args.img_h, args.img_w
cxy = torch.tensor((w/2.0, h/2.0), dtype=torch.float).cuda()
id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
model_3dmm = Face_3DMM(os.path.join(dir_path, '3DMM'),
                       id_dim, exp_dim, tex_dim, point_num)
lands_info = np.loadtxt(os.path.join(
    dir_path, '3DMM', 'lands_info.txt'), dtype=np.int32)
lands_info = torch.as_tensor(lands_info).cuda()
# mesh = openmesh.read_trimesh(os.path.join(dir_path, '3DMM', 'template.obj'))
focal = 1150

id_para = lms.new_zeros((1, id_dim), requires_grad=True)
exp_para = lms.new_zeros((num_frames, exp_dim), requires_grad=True)
tex_para = lms.new_zeros((1, tex_dim), requires_grad=True)
euler_angle = lms.new_zeros((num_frames, 3), requires_grad=True)
trans = lms.new_zeros((num_frames, 3), requires_grad=True)
light_para = lms.new_zeros((num_frames, 27), requires_grad=True)
trans.data[:, 2] -= 600
focal_length = lms.new_zeros(1, requires_grad=True)
focal_length.data += focal

set_requires_grad([id_para, exp_para, tex_para,
                   euler_angle, trans, light_para])

sel_ids = np.arange(0, num_frames, 10)
sel_num = sel_ids.shape[0]
arg_focal = 0.0
arg_landis = 1e5
for focal in range(500, 1500, 50):
    id_para = lms.new_zeros((1, id_dim), requires_grad=True)
    exp_para = lms.new_zeros((sel_num, exp_dim), requires_grad=True)
    euler_angle = lms.new_zeros((sel_num, 3), requires_grad=True)
    trans = lms.new_zeros((sel_num, 3), requires_grad=True)
    trans.data[:, 2] -= 600
    focal_length = lms.new_zeros(1, requires_grad=False)
    focal_length.data += focal
    set_requires_grad([id_para, exp_para, euler_angle, trans])

    optimizer_id = torch.optim.Adam([id_para], lr=.3)
    optimizer_exp = torch.optim.Adam([exp_para], lr=.3)
    optimizer_frame = torch.optim.Adam(
        [euler_angle, trans], lr=.3)
    iter_num = 2000

    for iter in range(iter_num):
        id_para_batch = id_para.expand(sel_num, -1)
        geometry = model_3dmm.forward_geo_sub(
            id_para_batch, exp_para, lands_info[-51:].long())
        proj_geo = forward_transform(
            geometry, euler_angle, trans, focal_length, cxy)
        loss_lan = cal_lan_loss(
            proj_geo[:, :, :2], lms[sel_ids, -51:, :].detach())
        loss_regid = torch.mean(id_para*id_para)*8
        loss_regexp = torch.mean(exp_para*exp_para)*0.5
        loss = loss_lan + loss_regid + loss_regexp
        optimizer_id.zero_grad()
        optimizer_exp.zero_grad()
        optimizer_frame.zero_grad()
        loss.backward()
        if iter > 1000:
            optimizer_id.step()
            optimizer_exp.step()
        optimizer_frame.step()
    print(focal, loss_lan.item(), torch.mean(trans[:, 2]).item())
    if loss_lan.item() < arg_landis:
        arg_landis = loss_lan.item()
        arg_focal = focal

sel_ids = np.arange(0, num_frames)
sel_num = sel_ids.shape[0]
id_para = lms.new_zeros((1, id_dim), requires_grad=True)
exp_para = lms.new_zeros((sel_num, exp_dim), requires_grad=True)
euler_angle = lms.new_zeros((sel_num, 3), requires_grad=True)
trans = lms.new_zeros((sel_num, 3), requires_grad=True)
trans.data[:, 2] -= 600
focal_length = lms.new_zeros(1, requires_grad=False)
focal_length.data += arg_focal
set_requires_grad([id_para, exp_para, euler_angle, trans])

optimizer_id = torch.optim.Adam([id_para], lr=.3)
optimizer_exp = torch.optim.Adam([exp_para], lr=.3)
optimizer_frame = torch.optim.Adam(
    [euler_angle, trans], lr=.3)
iter_num = 2000

for iter in range(iter_num):
    id_para_batch = id_para.expand(sel_num, -1)
    geometry = model_3dmm.forward_geo_sub(
        id_para_batch, exp_para, lands_info[-51:].long())
    proj_geo = forward_transform(
        geometry, euler_angle, trans, focal_length, cxy)
    loss_lan = cal_lan_loss(
        proj_geo[:, :, :2], lms[sel_ids, -51:, :].detach())
    loss_regid = torch.mean(id_para*id_para)*8
    loss_regexp = torch.mean(exp_para*exp_para)*0.5
    loss = loss_lan + loss_regid + loss_regexp
    optimizer_id.zero_grad()
    optimizer_exp.zero_grad()
    optimizer_frame.zero_grad()
    loss.backward()
    if iter > 1000:
        optimizer_id.step()
        optimizer_exp.step()
    optimizer_frame.step()
print(arg_focal, loss_lan.item(), torch.mean(trans[:, 2]).item())


torch.save({'id': id_para.detach().cpu(), 'exp': exp_para.detach().cpu(),
            'euler': euler_angle.detach().cpu(), 'trans': trans.detach().cpu(),
            'focal': focal_length.detach().cpu()}, os.path.join(os.path.dirname(args.path), 'track_params.pt'))
print('face tracking params saved')
