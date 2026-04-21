# CAPANet

This repository is adapted from the CATANet BasicSR codebase and now implements CAPANet for lightweight image super-resolution.

CAPANet keeps the efficient content-aware organization idea from CATANet, but replaces the original Token-Aggregation Block internals with three project modules:

- DPR: Dynamic Prototype Router
- PFSA: Progressive Focused Sparse Attention
- LMR: Low-to-High Multi-Level Reconstruction

The original CATANet paper and code remain the upstream reference:

### [[CATANet arXiv](https://arxiv.org/abs/2503.06896)] [[CATANet Supplementary Material](https://github.com/EquationWalker/CATANet/releases/tag/v0.1)] [[CATANet Pretrained Models](https://github.com/EquationWalker/CATANet/releases/tag/v0.0)] [[CATANet Visual Results](https://pan.quark.cn/s/f8ea09048957)]

## :newspaper:Upstream CATANet News

- :white_check_mark: 2025-03-15: Release the  [supplementary material](https://github.com/EquationWalker/CATANet/releases/tag/v0.1) of our CATANet.😃
- :white_check_mark: 2025-03-13: Release the  [pretrained models](https://github.com/EquationWalker/CATANet/releases/tag/v0.0)  and [visual results](https://pan.quark.cn/s/f8ea09048957) of our CATANet.🤗
- :white_check_mark: 2025-03-12:  arXiv paper available.
- :white_check_mark: 2025-03-09: Release the codes of our CATANet.
- :white_check_mark: 2025-02-28: Our CATANet was accepted by CVPR2025!:tada::tada::tada:

> CAPANet note: the current architecture is no longer a direct reproduction of upstream CATANet. The registered network type for new experiments is `CAPANet`; `CATANet` remains as a compatibility alias around the same implementation.

⭐If this work is helpful for you, please help star this repo. Thanks!🤗

## :bookmark_tabs:Contents
1. [Enviroment](#Environment)
1. [Training](#Training)
1. [Testing](#Testing)
1. [Citation](#Citation)
1. [Contact](#Contact)
1. [Acknowledgements](#Acknowledgements)


## :hammer:Environment
- Python 3.9
- PyTorch >=2.2

### Installation
```bash
pip install -r requirements.txt
python setup.py develop
```




## :rocket:Training
### Data Preparation
- Download the training dataset [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and put them in the folder `./datasets`.
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[Download](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing)]) and put them in the folder `./datasets`.
- It's recommended to refer to the data preparation from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) for faster data reading speed.

### Training Commands
- Refer to the training configuration files in `./options/train` folder for detailed settings.
```bash
# batch size = 4 (GPUs) × 16 (per GPU)
# training dataset: DIV2K

# ×2 scratch, input size = 64×64,800k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 basicsr/train.py -opt options/train/train_CATANet_x2_scratch.yml --launcher pytorch

# ×3 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 basicsr/train.py -opt options/train/train_CATANet_x3_finetune.yml --launcher pytorch

# ×4 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 basicsr/train.py -opt options/train/train_CATANet_x4_finetune.yml --launcher pytorch
```




## :wrench:Testing
### Data Preparation
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[Download](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing)]) and put them in the folder `./datasets`.

### Pretrained Models
- Upstream CATANet pretrained weights are not strictly compatible with the DPR + PFSA + LMR implementation because the block parameters and attention states have changed.
- Train CAPANet checkpoints with the configs in `options/train`, or load older CATANet checkpoints only with non-strict loading for partial warm start experiments.

### Testing Commands
- Refer to the testing configuration files in `./options/test` folder for detailed settings.


```bash
python basicsr/test.py -opt options/test/test_CATANet_x2.yml
python basicsr/test.py -opt options/test/test_CATANet_x3.yml
python basicsr/test.py -opt options/test/test_CATANet_x4.yml
```

## :kissing_heart:Citation

Please cite us if our work is useful for your research.

```
@article{liu2025CATANet,
  title={CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution},
  author={Xin Liu and Jie Liu and Jie Tang and Gangshan Wu},
  journal={arXiv preprint arXiv:2503.06896},
  year={2025}
}
```

## :mailbox:Contact

If you have any questions, feel free to approach me at xinliu2023@smail.nju.edu.cn

## 🥰Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).
