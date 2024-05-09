import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
from .network import AudioEncoder
import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_audio_features, get_rays, get_bg_coords, AudDataset

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def smooth_camera_path(poses, kernel_size=5):
    # smooth the camera trajectory...
    # poses: [N, 4, 4], numpy array

    N = poses.shape[0]
    K = kernel_size // 2
    
    trans = poses[:, :3, 3].copy() # [N, 3]
    rots = poses[:, :3, :3].copy() # [N, 3, 3]

    for i in range(N):
        start = max(0, i - K)
        end = min(N, i + K + 1)
        poses[i, :3, 3] = trans[start:end].mean(0)
        poses[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()

    return poses

def polygon_area(x, y):
    x_ = x - x.mean()
    y_ = y - y.mean()
    correction = x_[-1] * y_[0] - y_[-1]* x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5 * np.abs(main_area + correction)


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    print(f'[INFO] visualize poses: {poses.shape}')

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # 0 = disk, 1 = cpu, 2 = gpu
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16

        self.start_index = opt.data_range[0]
        self.end_index = opt.data_range[1]

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        # load nerf-compatible format data.
        
        # load all splits (train/valid/test)
        if type == 'all':
            transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
            transform = None
            for transform_path in transform_paths:
                with open(transform_path, 'r') as f:
                    tmp_transform = json.load(f)
                    if transform is None:
                        transform = tmp_transform
                    else:
                        transform['frames'].extend(tmp_transform['frames'])
        # load train and val split
        elif type == 'trainval':
            with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                transform = json.load(f)
            with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                transform_val = json.load(f)
            transform['frames'].extend(transform_val['frames'])
        # only load one specified split
        else:
            # no test, use val as test
            _split = 'val' if type == 'test' else type
            with open(os.path.join(self.root_path, f'transforms_{_split}.json'), 'r') as f:
                transform = json.load(f)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            self.H = int(transform['cy']) * 2 // downscale
            self.W = int(transform['cx']) * 2 // downscale
        
        # read images
        frames = transform["frames"]

        # use a slice of the dataset
        if self.end_index == -1: # abuse...
            self.end_index = len(frames)

        frames = frames[self.start_index:self.end_index]

        # use a subset of dataset.
        if type == 'train':
            if self.opt.part:
                frames = frames[::10] # 1/10 frames
            elif self.opt.part2:
                frames = frames[:375] # first 15s
        elif type == 'val':
            frames = frames[:100] # first 100 frames for val

        print(f'[INFO] load {len(frames)} {type} frames.')

        # only load pre-calculated aud features when not live-streaming
        if not self.opt.asr:

            # empty means the default self-driven extracted features.
            if self.opt.aud == '':
                if 'esperanto' in self.opt.asr_model:
                    aud_features = np.load(os.path.join(self.root_path, 'aud_eo.npy'))
                elif 'deepspeech' in self.opt.asr_model:
                    aud_features = np.load(os.path.join(self.root_path, 'aud_ds.npy'))
                # elif 'hubert_cn' in self.opt.asr_model:
                #     aud_features = np.load(os.path.join(self.root_path, 'aud_hu_cn.npy'))
                elif 'hubert' in self.opt.asr_model:
                    aud_features = np.load(os.path.join(self.root_path, 'aud_hu.npy'))
                elif self.opt.asr_model == 'ave':
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = AudioEncoder().to(device).eval()
                    ckpt = torch.load('./nerf_triplane/checkpoints/audio_visual_encoder.pth')
                    model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
                    dataset = AudDataset(os.path.join(self.root_path, 'aud.wav'))
                    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
                    outputs = []
                    for mel in data_loader:
                        mel = mel.to(device)
                        with torch.no_grad():
                            out = model(mel)
                        outputs.append(out)
                    outputs = torch.cat(outputs, dim=0).cpu()
                    first_frame, last_frame = outputs[:1], outputs[-1:]
                    aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)],
                                             dim=0).numpy()
                    # aud_features = np.load(os.path.join(self.root_path, 'aud_ave.npy'))
                else:
                    aud_features = np.load(os.path.join(self.root_path, 'aud.npy'))
            # cross-driven extracted features. 
            else:
                if self.opt.asr_model == 'ave':
                    try:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = AudioEncoder().to(device).eval()
                        ckpt = torch.load('./nerf_triplane/checkpoints/audio_visual_encoder.pth')
                        model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
                        dataset = AudDataset(self.opt.aud)
                        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
                        outputs = []
                        for mel in data_loader:
                            mel = mel.to(device)
                            with torch.no_grad():
                                out = model(mel)
                            outputs.append(out)
                        outputs = torch.cat(outputs, dim=0).cpu()
                        first_frame, last_frame = outputs[:1], outputs[-1:]
                        aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)], dim=0).numpy()
                    except:
                        print(f'[ERROR] If do not use Audio Visual Encoder, replace it with the npy file path.')
                else:
                    try:
                        aud_features = np.load(self.opt.aud)
                    except:
                        print(f'[ERROR] If do not use Audio Visual Encoder, replace it with the npy file path.')

            if self.opt.asr_model == 'ave':
                aud_features = torch.from_numpy(aud_features).unsqueeze(0)

                # support both [N, 16] labels and [N, 16, K] logits
                if len(aud_features.shape) == 3:
                    aud_features = aud_features.float().permute(1, 0, 2)  # [N, 16, 29] --> [N, 29, 16]

                    if self.opt.emb:
                        print(f'[INFO] argmax to aud features {aud_features.shape} for --emb mode')
                        aud_features = aud_features.argmax(1)  # [N, 16]

                else:
                    assert self.opt.emb, "aud only provide labels, must use --emb"
                    aud_features = aud_features.long()

                print(f'[INFO] load {self.opt.aud} aud_features: {aud_features.shape}')
            else:
                aud_features = torch.from_numpy(aud_features)

                # support both [N, 16] labels and [N, 16, K] logits
                if len(aud_features.shape) == 3:
                    aud_features = aud_features.float().permute(0, 2, 1)  # [N, 16, 29] --> [N, 29, 16]

                    if self.opt.emb:
                        print(f'[INFO] argmax to aud features {aud_features.shape} for --emb mode')
                        aud_features = aud_features.argmax(1)  # [N, 16]

                else:
                    assert self.opt.emb, "aud only provide labels, must use --emb"
                    aud_features = aud_features.long()

                print(f'[INFO] load {self.opt.aud} aud_features: {aud_features.shape}')

        if self.opt.au45:
            import pandas as pd
            au_blink_info = pd.read_csv(os.path.join(self.root_path, 'au.csv'))
            bs = au_blink_info[' AU45_r'].values
        else:
            bs = np.load(os.path.join(self.root_path, 'bs.npy'))
            if self.opt.bs_area == "upper":
                bs = np.hstack((bs[:, 0:5], bs[:, 8:10]))
            elif self.opt.bs_area == "single":
                bs = np.hstack((bs[:, 0].reshape(-1, 1),bs[:, 2].reshape(-1, 1),bs[:, 3].reshape(-1, 1), bs[:, 8].reshape(-1, 1)))
            elif self.opt.bs_area == "eye":
                bs = bs[:,8:10]


        self.torso_img = []
        self.images = []
        self.gt_images = []
        self.face_mask_imgs = []

        self.poses = []
        self.exps = []

        self.auds = []
        self.face_rect = []
        self.lhalf_rect = []
        self.upface_rect = []
        self.lowface_rect = []
        self.lips_rect = []
        self.eye_area = []
        self.eye_rect = []

        for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):

            f_path = os.path.join(self.root_path, 'gt_imgs', str(f['img_id']) + '.jpg')

            if not os.path.exists(f_path):
                print('[WARN]', f_path, 'NOT FOUND!')
                continue
            
            pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
            self.poses.append(pose)

            if self.preload > 0:
                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.images.append(image)
            else:
                self.images.append(f_path)

            if self.opt.portrait:
                gt_path = os.path.join(self.root_path, 'ori_imgs', str(f['img_id']) + '.jpg')
                # gt_path = os.path.join(self.root_path, 'torso_imgs', str(f['img_id']) + '_no_face.png')
                if not os.path.exists(f_path):
                    print('[WARN]', f_path, 'NOT FOUND!')
                    continue
                if self.preload > 0:
                    gt_image = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
                    gt_image = gt_image.astype(np.float32) / 255 # [H, W, 3/4]

                    self.gt_images.append(gt_image)
                else:
                    self.gt_images.append(gt_path)

                face_mask_path = os.path.join(self.root_path, 'parsing', str(f['img_id']) + '_face.png')
                if not os.path.exists(face_mask_path):
                    print('[WARN]', face_mask_path, 'NOT FOUND!')
                    continue
                if self.preload > 0:
                    face_mask_img = (255 - cv2.imread(face_mask_path)[:, :, 1]) / 255.0
                    self.face_mask_imgs.append(face_mask_img)
                else:
                    self.face_mask_imgs.append(face_mask_path)

            # load frame-wise bg
        
            torso_img_path = os.path.join(self.root_path, 'torso_imgs', str(f['img_id']) + '.png')

            if self.preload > 0:
                torso_img = cv2.imread(torso_img_path, cv2.IMREAD_UNCHANGED) # [H, W, 4]
                torso_img = cv2.cvtColor(torso_img, cv2.COLOR_BGRA2RGBA)
                torso_img = torso_img.astype(np.float32) / 255 # [H, W, 3/4]

                self.torso_img.append(torso_img)
            else:
                self.torso_img.append(torso_img_path)

            # find the corresponding audio to the image frame
            if not self.opt.asr and self.opt.aud == '':
                aud = aud_features[min(f['aud_id'], aud_features.shape[0] - 1)] # careful for the last frame...
                self.auds.append(aud)

            # load lms and extract face
            lms = np.loadtxt(os.path.join(self.root_path, 'ori_imgs', str(f['img_id']) + '.lms')) # [68, 2]

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            upface_xmin, upface_xmax = int(lms[:, 1].min()),int(lms[30,1])
            lowface_xmin, lowface_xmax = int(lms[30,1]), int(lms[:, 1].max())
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            self.face_rect.append([xmin, xmax, ymin, ymax])
            self.lhalf_rect.append([lh_xmin, lh_xmax, ymin, ymax])
            self.upface_rect.append([upface_xmin, upface_xmax, ymin, ymax])
            self.lowface_rect.append([lowface_xmin, lowface_xmax, ymin, ymax])


            if self.opt.exp_eye:
                area = bs[f['img_id']]
                if self.opt.au45:
                    area = np.clip(area, 0, 2) / 2
                self.eye_area.append(area)

                xmin, xmax = int(lms[36:48, 1].min()), int(lms[36:48, 1].max())
                ymin, ymax = int(lms[36:48, 0].min()), int(lms[36:48, 0].max())
                self.eye_rect.append([xmin, xmax, ymin, ymax])

            if self.opt.finetune_lips:
                lips = slice(48, 60)
                xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
                ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

                # padding to H == W
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2

                l = max(xmax - xmin, ymax - ymin) // 2
                xmin = max(0, cx - l)
                xmax = min(self.H, cx + l)
                ymin = max(0, cy - l)
                ymax = min(self.W, cy + l)

                self.lips_rect.append([xmin, xmax, ymin, ymax])
        
        # load pre-extracted background image (should be the same size as training image...)

        if self.opt.bg_img == 'white': # special
            bg_img = np.ones((self.H, self.W, 3), dtype=np.float32)
        elif self.opt.bg_img == 'black': # special
            bg_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        else: # load from file
            # default bg
            if self.opt.bg_img == '':
                self.opt.bg_img = os.path.join(self.root_path, 'bc.jpg')
            bg_img = cv2.imread(self.opt.bg_img, cv2.IMREAD_UNCHANGED) # [H, W, 3]
            if bg_img.shape[0] != self.H or bg_img.shape[1] != self.W:
                bg_img = cv2.resize(bg_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = bg_img.astype(np.float32) / 255 # [H, W, 3/4]

        self.bg_img = bg_img

        self.poses = np.stack(self.poses, axis=0)

        # smooth camera path...
        if self.opt.smooth_path:
            self.poses = smooth_camera_path(self.poses, self.opt.smooth_path_window)
            
        self.poses = torch.from_numpy(self.poses) # [N, 4, 4]

        if self.preload > 0:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
            self.torso_img = torch.from_numpy(np.stack(self.torso_img, axis=0)) # [N, H, W, C]
            if self.opt.portrait:
                self.gt_images = torch.from_numpy(np.stack(self.gt_images, axis=0)) # [N, H, W, C]
                self.face_mask_imgs = torch.from_numpy(np.stack(self.face_mask_imgs, axis=0)) # [N, H, W, C]

        else:
            self.images = np.array(self.images)
            self.torso_img = np.array(self.torso_img)
            if self.opt.portrait:
                self.gt_images = np.array(self.gt_images)
                self.face_mask_imgs = np.array(self.face_mask_imgs)


        if self.opt.asr:
            # live streaming, no pre-calculated auds
            self.auds = None
        else:
            # auds corresponding to images
            if self.opt.aud == '':
                self.auds = torch.stack(self.auds, dim=0) # [N, 32, 16]
            # auds is novel, may have a different length with images
            else:
                self.auds = aud_features
        
        self.bg_img = torch.from_numpy(self.bg_img)

        if self.opt.exp_eye:
            self.eye_area = np.array(self.eye_area, dtype=np.float32) # [N]
            print(f'[INFO] eye_area: {self.eye_area.min()} - {self.eye_area.max()}')

            if self.opt.smooth_eye:

                # naive 5 window average
                ori_eye = self.eye_area.copy()
                for i in range(ori_eye.shape[0]):
                    start = max(0, i - 1)
                    end = min(ori_eye.shape[0], i + 2)
                    self.eye_area[i] = ori_eye[start:end].mean()
            if self.opt.au45:
                self.eye_area = torch.from_numpy(self.eye_area).view(-1, 1)  # [N, 1]
            else:
                if self.opt.bs_area == "upper":
                    self.eye_area = torch.from_numpy(self.eye_area).view(-1, 7) # [N, 7]
                elif self.opt.bs_area == "single":
                    self.eye_area = torch.from_numpy(self.eye_area).view(-1, 4)  # [N, 7]
                else:
                    self.eye_area = torch.from_numpy(self.eye_area).view(-1, 2)
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        
        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload > 1:
            self.poses = self.poses.to(self.device)

            if self.auds is not None:
                self.auds = self.auds.to(self.device)

            self.bg_img = self.bg_img.to(torch.half).to(self.device)

            self.torso_img = self.torso_img.to(torch.half).to(self.device)
            self.images = self.images.to(torch.half).to(self.device)
            if self.opt.portrait:
                self.gt_images = self.gt_images.to(torch.half).to(self.device)
                self.face_mask_imgs = self.face_mask_imgs.to(torch.half).to(self.device)
            
            if self.opt.exp_eye:
                self.eye_area = self.eye_area.to(self.device)

        # load intrinsics
        if 'focal_len' in transform:
            fl_x = fl_y = transform['focal_len']
        elif 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        # directly build the coordinate meshgrid in [-1, 1]^2
        self.bg_coords = get_bg_coords(self.H, self.W, self.device) # [1, H*W, 2] in [-1, 1]


    def mirror_index(self, index):
        size = self.poses.shape[0]
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1


    def collate(self, index):

        B = len(index) # a list of length 1
        # assert B == 1

        results = {}

        # audio use the original index
        if self.auds is not None:
            auds = get_audio_features(self.auds, self.opt.att, index[0]).to(self.device)
            results['auds'] = auds

        # head pose and bg image may mirror (replay --> <-- --> <--).
        index[0] = self.mirror_index(index[0])

        poses = self.poses[index].to(self.device) # [B, 4, 4]
        
        if self.training and self.opt.finetune_lips:
            rect = self.lips_rect[index[0]]
            results['rect'] = rect
            rays = get_rays(poses, self.intrinsics, self.H, self.W, -1, rect=rect)
        else:
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, self.opt.patch_size)
        results['up_rect'] = self.upface_rect[index[0]]
        results['low_rect'] = self.lowface_rect[index[0]]
        results['index'] = index # for ind. code
        results['H'] = self.H
        results['W'] = self.W
        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']

        # get a mask for rays inside rect_face
        if self.training:
            xmin, xmax, ymin, ymax = self.face_rect[index[0]]
            face_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax) # [B, N]
            results['face_mask'] = face_mask
            
            xmin, xmax, ymin, ymax = self.lhalf_rect[index[0]]
            lhalf_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax) # [B, N]
            results['lhalf_mask'] = lhalf_mask

            xmin, xmax, ymin, ymax = self.upface_rect[index[0]]
            upface_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax) # [B, N]
            results['upface_mask'] = upface_mask

            xmin, xmax, ymin, ymax = self.lowface_rect[index[0]]
            lowface_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax) # [B, N]
            results['lowface_mask'] = lowface_mask


        if self.opt.exp_eye:
            results['eye'] = self.eye_area[index].to(self.device) # [1]
            if self.training:
                #results['eye'] += (np.random.rand()-0.5) / 10
                xmin, xmax, ymin, ymax = self.eye_rect[index[0]]
                eye_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax) # [B, N]
                results['eye_mask'] = eye_mask

        else:
            results['eye'] = None

        # load bg
        bg_torso_img = self.torso_img[index]
        if self.preload == 0: # on the fly loading
            bg_torso_img = cv2.imread(bg_torso_img[0], cv2.IMREAD_UNCHANGED) # [H, W, 4]
            bg_torso_img = cv2.cvtColor(bg_torso_img, cv2.COLOR_BGRA2RGBA)
            bg_torso_img = bg_torso_img.astype(np.float32) / 255 # [H, W, 3/4]
            bg_torso_img = torch.from_numpy(bg_torso_img).unsqueeze(0)
        bg_torso_img = bg_torso_img[..., :3] * bg_torso_img[..., 3:] + self.bg_img * (1 - bg_torso_img[..., 3:])
        bg_torso_img = bg_torso_img.view(B, -1, 3).to(self.device)

        if not self.opt.torso:
            bg_img = bg_torso_img
        else:
            bg_img = self.bg_img.view(1, -1, 3).repeat(B, 1, 1).to(self.device)

        if self.training:
            bg_img = torch.gather(bg_img, 1, torch.stack(3 * [rays['inds']], -1)) # [B, N, 3]

        results['bg_color'] = bg_img

        if self.opt.torso and self.training:
            bg_torso_img = torch.gather(bg_torso_img, 1, torch.stack(3 * [rays['inds']], -1)) # [B, N, 3]
            results['bg_torso_color'] = bg_torso_img

        if self.opt.portrait:
            bg_gt_images = self.gt_images[index]
            if self.preload == 0:
                bg_gt_images = cv2.imread(bg_gt_images[0], cv2.IMREAD_UNCHANGED)
                bg_gt_images = cv2.cvtColor(bg_gt_images, cv2.COLOR_BGR2RGB)
                bg_gt_images = bg_gt_images.astype(np.float32) / 255
                bg_gt_images = torch.from_numpy(bg_gt_images).unsqueeze(0)
            bg_gt_images = bg_gt_images.to(self.device)
            results['bg_gt_images'] = bg_gt_images

            bg_face_mask = self.face_mask_imgs[index]
            if self.preload == 0:
                # bg_face_mask = np.all(cv2.imread(bg_face_mask[0]) == [255, 0, 0], axis=-1).astype(np.uint8)
                bg_face_mask = (255 - cv2.imread(bg_face_mask[0])[:, :, 1]) / 255.0
                bg_face_mask = torch.from_numpy(bg_face_mask).unsqueeze(0)
            bg_face_mask = bg_face_mask.to(self.device)
            results['bg_face_mask'] = bg_face_mask


        images = self.images[index] # [B, H, W, 3/4]
        if self.preload == 0:
            images = cv2.imread(images[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            images = images.astype(np.float32) / 255 # [H, W, 3]
            images = torch.from_numpy(images).unsqueeze(0)
        images = images.to(self.device)

        if self.training:
            C = images.shape[-1]
            images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
        results['images'] = images

        if self.training:
            bg_coords = torch.gather(self.bg_coords, 1, torch.stack(2 * [rays['inds']], -1)) # [1, N, 2]
        else:
            bg_coords = self.bg_coords # [1, N, 2]

        results['bg_coords'] = bg_coords

        # results['poses'] = convert_poses(poses) # [B, 6]
        # results['poses_matrix'] = poses # [B, 4, 4]
        results['poses'] = poses # [B, 4, 4]
            
        return results

    def dataloader(self):

        if self.training:
            # training len(poses) == len(auds)
            size = self.poses.shape[0]
        else:
            # test with novel auds, then use its length
            if self.auds is not None:
                size = self.auds.shape[0]
            # live stream test, use 2 * len(poses), so it naturally mirrors.
            else:
                size = 2 * self.poses.shape[0]

        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need poses in trainer.

        # do evaluate if has gt images and use self-driven setting
        loader.has_gt = (self.opt.aud == '')

        return loader        
