# SyncTalk: The Devil😈 is in the Synchronization for Talking Head Synthesis [CVPR 2024]
The official repository of the paper [SyncTalk: The Devil is in the Synchronization for Talking Head Synthesis](https://arxiv.org/abs/2311.17590)

<p align='center'>
  <b>
    <a href="https://arxiv.org/abs/2311.17590">Paper</a>
    | 
    <a href="https://ziqiaopeng.github.io/synctalk/">Project Page</a>
    |
    <a href="https://github.com/ZiqiaoPeng/SyncTalk">Code</a> 
  </b>
</p> 

Colab notebook demonstration: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egq0_ZK5sJAAawShxC0y4JRZQuVS2X-Z?usp=sharing)

A short demo video can be found [here](./demo/short_demo.mp4).

  <p align='center'>  
    <img src='assets/image/synctalk.png' width='1000'/>
  </p>

  The proposed **SyncTalk** synthesizes synchronized talking head videos, employing tri-plane hash representations to maintain subject identity. It can generate synchronized lip movements, facial expressions, and stable head poses, and restores hair details to create high-resolution videos.

## 🔥🔥🔥 News
- [2023-11-30] Update arXiv paper.
- [2024-03-04] The code and pre-trained model are released.
- [2024-03-22] The Google Colab notebook is released.
- [2024-04-14] Add Windows support.
- [2024-04-28] The preprocessing code is released.
- [2024-04-29] Fix bugs: audio encoder, blendshape capture, and face tracker.
- [2024-05-03] Try replacing NeRF with Gaussian Splatting. Code: [GS-SyncTalk](https://github.com/ZiqiaoPeng/GS-SyncTalk)
- **[2024-05-24] Introduce torso training to repair double chin.**



## For Windows
Thanks to [okgpt](https://github.com/okgptai), we have launched a Windows integration package, you can download `SyncTalk-Windows.zip` and unzip it, double-click `inference.bat` to run the demo.

Download link: [Hugging Face](https://huggingface.co/ZiqiaoPeng/SyncTalk/blob/main/SyncTalk-Windows.zip) ||  [Baidu Netdisk](https://pan.baidu.com/s/1g3312mZxx__T6rAFPHjrRg?pwd=6666)

## For Linux

### Installation

Tested on Ubuntu 18.04, Pytorch 1.12.1 and CUDA 11.3.
```bash
git clone https://github.com/ZiqiaoPeng/SyncTalk.git
cd SyncTalk
```
#### Install dependency

```bash
conda create -n synctalk python==3.8.8
conda activate synctalk
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
sudo apt-get install portaudio19-dev
pip install -r requirements.txt
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html
pip install tensorflow-gpu==2.8.1
pip install ./freqencoder
pip install ./shencoder
pip install ./gridencoder
pip install ./raymarching
```
If you encounter problems installing PyTorch3D, you can use the following command to install it:
```bash
python ./scripts/install_pytorch3d.py
```

### Data Preparation
#### Pre-trained model
Please place the [May.zip](https://drive.google.com/file/d/18Q2H612CAReFxBd9kxr-i1dD8U1AUfsV/view?usp=sharing) in the **data** folder, the [trial_may.zip](https://drive.google.com/file/d/1C2639qi9jvhRygYHwPZDGs8pun3po3W7/view?usp=sharing) in the **model** folder, and then unzip them.
#### [New] Process your video
- Prepare face-parsing model.

  ```bash
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth
  ```

- Prepare the 3DMM model for head pose estimation.

  ```bash
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/exp_info.npy?raw=true -O data_utils/face_tracking/3DMM/exp_info.npy
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/keys_info.npy?raw=true -O data_utils/face_tracking/3DMM/keys_info.npy
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/sub_mesh.obj?raw=true -O data_utils/face_tracking/3DMM/sub_mesh.obj
  wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/topology_info.npy?raw=true -O data_utils/face_tracking/3DMM/topology_info.npy
  ```

- Download 3DMM model from [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details):

  ```
  # 1. copy 01_MorphableModel.mat to data_util/face_tracking/3DMM/
  # 2.
    cd data_utils/face_tracking
    python convert_BFM.py
  ```
- Put your video under `data/<ID>/<ID>.mp4`, and then run the following command to process the video.
  
  **[Note]** The video must be 25FPS, with all frames containing the talking person. The resolution should be about 512x512, and duration about 4-5 min.
  ```bash
  python data_utils/process.py data/<ID>/<ID>.mp4 --asr ave
  ```
  You can choose to use AVE, DeepSpeech or Hubert. The processed video will be saved in the **data** folder. 


- [Optional] Obtain AU45 for eyes blinking
  
  Run `FeatureExtraction` in [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), rename and move the output CSV file to `data/<ID>/au.csv`.


  **[Note]** Since EmoTalk's blendshape capture is not open source, the preprocessing code here is replaced with mediapipe's blendshape capture. But according to some feedback, it doesn't work well, you can choose to replace it with AU45. If you want to compare with SyncTalk, some results from using EmoTalk capture can be obtained [here](https://drive.google.com/drive/folders/1LLFtQa2Yy2G0FaNOxwtZr0L974TXCYKh?usp=sharing) and videos from [GeneFace](https://drive.google.com/drive/folders/1vimGVNvP6d6nmmc8yAxtWuooxhJbkl68).


### Quick Start
#### Run the evaluation code
```bash
python main.py data/May --workspace model/trial_may -O --test --asr_model ave

python main.py data/May --workspace model/trial_may -O --test --asr_model ave --portrait
```
“ave” refers to our Audio Visual Encoder, “portrait” signifies pasting the generated face back onto the original image, representing higher quality.

If it runs correctly, you will get the following results.

| Setting                  | PSNR   | LPIPS  | LMD   |
|--------------------------|--------|--------|-------|
| SyncTalk (w/o Portrait)  | 32.201 | 0.0394 | 2.822 |
| SyncTalk (Portrait)      | 37.644 | 0.0117 | 2.825 |

This is for a single subject; the paper reports the average results for multiple subjects.

#### Inference with target audio
```bash
python main.py data/May --workspace model/trial_may -O --test --test_train --asr_model ave --portrait --aud ./demo/test.wav
```
Please use files with the “.wav” extension for inference, and the inference results will be saved in “model/trial_may/results/”. If do not use Audio Visual Encoder, replace wav with the npy file path.
* DeepSpeech

  ```bash
  python data_utils/deepspeech_features/extract_ds_features.py --input data/<name>.wav # save to data/<name>.npy
  ```
* HuBERT

  ```bash
  # Borrowed from GeneFace. English pre-trained.
  python data_utils/hubert.py --wav data/<name>.wav # save to data/<name>_hu.npy
  ```
### Train
```bash
# by default, we load data from disk on the fly.
# we can also preload all data to CPU/GPU for faster training, but this is very memory-hungry for large datasets.
# `--preload 0`: load from disk (default, slower).
# `--preload 1`: load to CPU (slightly slower)
# `--preload 2`: load to GPU (fast)
python main.py data/May --workspace model/trial_may -O --iters 60000 --asr_model ave
python main.py data/May --workspace model/trial_may -O --iters 100000 --finetune_lips --patch_size 64 --asr_model ave

# or you can use the script to train
sh ./scripts/train_may.sh
```
**[Tips]** Audio visual encoder (AVE) is suitable for characters with accurate lip sync and large lip movements such as May and Shaheen. Using AVE in the inference stage can achieve more accurate lip sync. If your training results show lip jitter, please try using deepspeech or hubert model as audio feature encoder. 

```bash
# Use deepspeech model
python main.py data/May --workspace model/trial_may -O --iters 60000 --asr_model deepspeech
python main.py data/May --workspace model/trial_may -O --iters 100000 --finetune_lips --patch_size 64 --asr_model deepspeech

# Use hubert model
python main.py data/May --workspace model/trial_may -O --iters 60000 --asr_model hubert
python main.py data/May --workspace model/trial_may -O --iters 100000 --finetune_lips --patch_size 64 --asr_model hubert
```

If you want to use the OpenFace au45 as the eye parameter, please add "--au45" to the command line.

```bash
# Use OpenFace AU45
python main.py data/May --workspace model/trial_may -O --iters 60000 --asr_model ave --au45
python main.py data/May --workspace model/trial_may -O --iters 100000 --finetune_lips --patch_size 64 --asr_model ave --au45
```

### Test
```bash
python main.py data/May --workspace model/trial_may -O --test --asr_model ave --portrait

```

### Train & Test Torso [Repair Double Chin]
If your character trained only the head appeared double chin problem, you can introduce torso training. By training the torso, this problem can be solved, but **you will not be able to use the "--portrait" mode.** If you add "--portrait", the torso model will fail!

```bash
# Train
# <head>.pth should be the latest checkpoint in trial_may
python main.py data/May/ --workspace model/trial_may_torso/ -O --torso --head_ckpt <head>.pth --iters 150000 --asr_model ave

# For example
python main.py data/May/ --workspace model/trial_may_torso/ -O --torso --head_ckpt model/trial_may/ngp_ep0019.pth --iters 150000 --asr_model ave

# Test
python main.py data/May --workspace model/trial_may_torso -O  --torso --test --asr_model ave  # not support --portrait

# Inference with target audio
python main.py data/May --workspace model/trial_may_torso -O  --torso --test --test_train --asr_model ave --aud ./demo/test.wav # not support --portrait

```



## Citation	

```
@InProceedings{peng2023synctalk,
  title     = {SyncTalk: The Devil is in the Synchronization for Talking Head Synthesis}, 
  author    = {Ziqiao Peng and Wentao Hu and Yue Shi and Xiangyu Zhu and Xiaomei Zhang and Jun He and Hongyan Liu and Zhaoxin Fan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
}
```

## Acknowledgement
This code is developed heavily relying on [ER-NeRF](https://github.com/Fictionarry/ER-NeRF), and also [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF), [GeneFace](https://github.com/yerfor/GeneFace), [DFRF](https://github.com/sstzal/DFRF), [DFA-NeRF](https://github.com/ShunyuYao/DFA-NeRF/), [AD-NeRF](https://github.com/YudongGuo/AD-NeRF), and [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch).

Thanks for these great projects. Thanks to [Tiandishihua](https://github.com/Tiandishihua) for helping us fix the bug that loss equals NaN.

## Disclaimer
By using the "SyncTalk", users agree to comply with all applicable laws and regulations, and acknowledge that misuse of the software, including the creation or distribution of harmful content, is strictly prohibited. The developers of the software disclaim all liability for any direct, indirect, or consequential damages arising from the use or misuse of the software.
