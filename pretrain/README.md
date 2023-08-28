# Pretraining of GeoMIM

The pretraining is implemented with the [bevfusion](https://github.com/mit-han-lab/bevfusion) framework.
We use the LiDAR-only model in [bevfusion](https://github.com/mit-han-lab/bevfusion) (69.28 NDS) for pretraining. 

## Pretrained Models
We provide pretrained models that have been trained on the NuScenes dataset. These models can be used as a starting point for your own tasks or fine-tuning.

| Config     | Epoch | Download |	
| ---------- | ----- | ----- |
| [Swin-Base](geomim_config/pretrain_base_50ep.yaml)  |   50  | [Model](https://drive.google.com/file/d/1bHp4Y4j8X4LRoi8n7PRYskFohYq_J2X8/view?usp=sharing) |
| [Swin-Large](geomim_config/pretrain_large_50ep.yaml) |   50  | [Model](https://drive.google.com/file/d/1lMdkAV0MTbBkR4YzsyPPGg0EkB5vLTVG/view?usp=sharing) |

## Usage

### Installation

The code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)

After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```

### Data Preparation

#### nuScenes

Please follow the instructions from [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) to download and preprocess the nuScenes dataset. After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```

### LiDAR and MixMAE models setups

To set up the LiDAR model for pretraining, follow these steps:

Download the [LiDAR model](https://bevfusion.mit.edu/files/pretrained/lidar-only-det.pth) weights from [bevfusion](https://github.com/mit-han-lab/bevfusion).

Download the [Swin-Base](https://drive.google.com/file/d/1pZYmTv08xK_kOe2kk6ahuvgJVkHm-ZIa/view?usp=sharing)/[Swin-Large](https://drive.google.com/file/d/1dM8Lu2nVEukxPwn7PLmDmRAYwQV59ttx/view?usp=sharing) weights from [MixMAE](https://github.com/Sense-X/MixMIM).

### Pre-Training

We provide instructions to reproduce our results on nuScenes. You can use pytorch or slurm for distributed training. 

For example, the Swin-Base model can be pretrained with:


```bash
sh run_pretrain.sh partition 8 config/pretrain_base_50ep.yaml runs/pretrain/pretrain_base_50ep
```

The large model can be pretrained as the same. 

## Acknowledgements

The pretraining code is based on [bevfusion](https://github.com/mit-han-lab/bevfusion). 
