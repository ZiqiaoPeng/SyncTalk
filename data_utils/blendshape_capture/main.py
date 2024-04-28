# -*-coding:utf-8-*-
import argparse
import os
import random
import numpy as np
import cv2
import glob
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
import onnxruntime as ort
from collections import OrderedDict
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from tqdm import tqdm
from facenet_pytorch import MTCNN
#from mtcnn_ort import MTCNN


def infer_bs(root_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_options = python.BaseOptions(model_asset_path="./data_utils/blendshape_capture/face_landmarker.task")
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    for i in os.listdir(root_path):
        if i.endswith(".mp4"):
            mp4_path = os.path.join(root_path, i)
            npy_path = os.path.join(root_path, "bs_mp.npy")
            if os.path.exists(npy_path):
                print("npy file exists:", i.split(".")[0])
                continue
            else:
                print("npy file not exists:", i.split(".")[0])
                image_path = os.path.join(root_path, "img/temp.png")
                os.makedirs(os.path.join(root_path, 'img/'), exist_ok=True)
                cap = cv2.VideoCapture(mp4_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print("fps:", fps)
                print("frame_count:", frame_count)
                k = 0
                total = frame_count
                bs = np.zeros((int(total), 52), dtype=np.float32)
                print("total:", total)
                print("videoPath:{} fps:{},k".format(mp4_path.split('/')[-1], fps))
                pbar = tqdm(total=int(total))
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(image_path, frame)
                        image = mp.Image.create_from_file(image_path)
                        result = detector.detect(image)
                        face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in
                                                   result.face_blendshapes[0]]
                        blendshape_coef = np.array(face_blendshapes_scores)[1:]
                        blendshape_coef = np.append(blendshape_coef, 0)
                        bs[k] = blendshape_coef
                        pbar.update(1)
                        k += 1
                    else:
                        break
                cap.release()
                pbar.close()
                # np.save(npy_path, bs)
                # print(np.shape(bs))
                output = np.zeros((bs.shape[0], bs.shape[1]))
                for j in range(bs.shape[1]):
                    output[:, j] = savgol_filter(bs[:, j], 5, 3)
                np.save(npy_path, output)
                print(np.shape(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="idname of target person")
    args = parser.parse_args()
    infer_bs(args.path)