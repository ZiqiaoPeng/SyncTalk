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

  <p align='center'>  
    <img src='assets/image/synctalk.png' width='1000'/>
  </p>

  The proposed **SyncTalk** synthesizes synchronized talking head videos, employing tri-plane hash representations to maintain subject identity. It can generate synchronized lip movements, facial expressions, and stable head poses, and restores hair details to create high-resolution videos.

## Installation

Tested on Ubuntu 18.04, Pytorch 1.12.1 and CUDA 11.3.
```bash
git clone https://github.com/ZiqiaoPeng/SyncTalk.git
cd SyncTalk
```
### Install dependency

```bash
conda create -n synctalk python==3.8.8
conda activate synctalk
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html
pip install ./freqencoder
pip install ./shencoder
pip install ./gridencoder
pip install ./raymarching
```
If you encounter problems installing PyTorch3D, you can use the following command to install it:
```bash
python ./scripts/install_pytorch3d.py
```

## Data Preparation
Please place the [May.zip](https://drive.google.com/file/d/18Q2H612CAReFxBd9kxr-i1dD8U1AUfsV/view?usp=sharing) in the **data** folder, the [trial_may.zip](https://drive.google.com/file/d/1C2639qi9jvhRygYHwPZDGs8pun3po3W7/view?usp=sharing) in the **model** folder, and then unzip them.

## Quick Start
### Run the evaluation code
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

### Inference with target audio
```bash
python main.py data/May --workspace model/trial_may -O --test --test_train --asr_model ave --portrait --aud ./demo/test.wav
```
Please use files with the “.wav” extension for inference, and the inference results will be saved in “model/trial_may/results/”.
## Train
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

## Test
```bash
python main.py data/May --workspace model/trial_may -O --test --asr_model ave --portrait
```


## TODO
- [x] **Release Training Code.**
- [x] **Release Pre-trained Model.**
- [x] **Release Google Colab.**
- [ ] Release Preprocessing Code.



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
This code is developed heavily relying on [ER-NeRF](https://github.com/Fictionarry/ER-NeRF), and also [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF), [GeneFace](https://github.com/yerfor/GeneFace), [DFRF](https://github.com/sstzal/DFRF), [AD-NeRF](https://github.com/YudongGuo/AD-NeRF), and [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch).

Thanks for these great projects.
