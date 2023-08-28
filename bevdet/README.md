# BEVDet finetuning with GeoMIM

The finetuning is implemented with the [BEVDet](https://github.com/HuangJunJie2017/BEVDet) framework. We also provide models trained with its advanced setups. 

## Main Results

### Nuscenes Detection

#### Results from GeoMIM
| Config | mAP | NDS | Download |	
| ------ | --- | --- | ----- |
| [bevdet-swinb-4d-256x704-cbgs](configs/bevdet_geomim/bevdet-swinb-4d-256x704-cbgs.py) | 33.98 | 47.19 | [Model](https://drive.google.com/file/d/1sNn6kdgdIUQwUZQvywvtszEHTrStsDFQ/view?usp=sharing) |
| [bevdet-swinb-4d-256x704-cbgs-geomim](configs/bevdet_geomim/bevdet-swinb-4d-256x704-cbgs-geomim.py) | 42.25 | 53.1 | [Model](https://drive.google.com/file/d/1xJ77En6Aa_gYDu1IEAtx_t8gdS3MCici/view?usp=sharing) |
| [bevdet-swinb-4d-stereo-256x704-cbgs-geomim](configs/bevdet_geomim/bevdet-swinb-4d-stereo-256x704-cbgs-geomim.py) | 45.33 | 55.1 | [Model](https://drive.google.com/file/d/1v-YXOJ-VLuCS-WrnJHfmA_YKgWshjMe2/view?usp=sharing) |
| [bevdet-swinb-4d-stereo-512x1408-cbgs-geomim](configs/bevdet_geomim/bevdet-swinb-4d-stereo-512x1408-cbgs-geomim.py) | 52.04 | 60.92 | [Model](https://drive.google.com/file/d/1AZTjIrO0G1huecHx5PzJNgReVGCo1JdC/view?usp=sharing) |


Note that all the results are obtained by aligning previous frame bev feature during the view transformation.

#### Results from original BEVDet
| Config                                                                    | mAP        | NDS        | Latency(ms) | FPS  | Model                                                                                          | Log                                                                                            |
| ------------------------------------------------------------------------- | ---------- | ---------- | ---- | ---- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [**BEVDet-R50**](configs/bevdet/bevdet-r50.py)                            | 28.3       | 35.0       | 29.1/4.2/33.3| 30.7 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-CBGS**](configs/bevdet/bevdet-r50-cbgs.py)                  | 31.3       | 39.8       |28.9/4.3/33.2 |30.1 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-4D-CBGS**](configs/bevdet/bevdet-r50-4d-cbgs.py) | 31.4/35.4# | 44.7/44.9# | 29.1/4.3/33.4|30.0 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |[baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1)|
| [**BEVDet-R50-4D-Depth-CBGS**](configs/bevdet/bevdet-r50-4d-depth-cbgs.py) | 36.1/36.2# | 48.3/48.4# |35.7/4.0/39.7 |25.2 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-4D-Stereo-CBGS**](configs/bevdet/bevdet-r50-4d-stereo-cbgs.py) | 38.2/38.4# | 49.9/50.0# |-  |-  | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-4DLongterm-CBGS**](configs/bevdet/bevdet-r50-4dlongterm-cbgs.py) | 34.8/35.4# | 48.2/48.7# | 30.8/4.2/35.0|28.6 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-4DLongterm-Depth-CBGS**](configs/bevdet/bevdet-r50-4d-depth-cbgs.py) | 39.4/39.9# | 51.5/51.9# |38.4/4.0/42.4 |23.6 | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-R50-4DLongterm-Stereo-CBGS**](configs/bevdet/bevdet-r50-4dlongterm-stereo-cbgs.py) | 41.1/41.5# | 52.3/52.7# |- |- | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-STBase-4D-Stereo-512x1408-CBGS**](configs/bevdet/bevdet-stbase-4d-stereo-512x1408-cbgs.py) | 47.2# | 57.6# |-  |-  | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |

\# align previous frame bev feature during the view transformation.

Depth: Depth supervised from Lidar as BEVDepth.

Longterm: cat 8 history frame in temporal modeling. 1 by default. 

Stereo: A private implementation that concat cost-volumn with image feature before executing model.view_transformer.depth_net.

The latency includes Network/Post-Processing/Total. Training without CBGS is deprecated.


### Nuscenes Occupancy


#### Results from GeoMIM

| Config | mIoU | Download |	
| ------ | ---- | ----- |
| [bevdet-occ-swinb-4d-stereo-2x-geomim](configs/bevdet_geomim/bevdet-occ-swinb-4d-stereo-512x1408-24e-geomim.py) | 45.0 | [Model](https://drive.google.com/file/d/1qH5UalLpXueglGhEfCo58hi85LBilSMa/view?usp=sharing) |
| [bevdet-occ-swinb-4d-stereo-2x-geomim](configs/bevdet_geomim/bevdet-occ-swinb-4d-stereo-512x1408-24e-geomim-load.py) (*) | 45.73 | [Model](https://drive.google.com/file/d/11Qi8BgJaPI4YU4q1XQ963WvwBYqx_EbY/view?usp=sharing) |
| [bevdet-occ-swinl-4d-stereo-2x-geomim](configs/bevdet_geomim/bevdet-occ-swinl-4d-stereo-512x1408-24e-geomim.py) | 46.27 | [Model](https://drive.google.com/file/d/1tqb_CE4tIN1tsuD4tEimdurFXbLsMzzB/view?usp=sharing) |


#### Results from original BEVDet

| Config                                                                    | mIOU       | Model | Log                                                                                            |
| ------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [**BEVDet-Occ-R50-4D-Stereo-2x**](configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e.py)                                 | 36.1     | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-Occ-R50-4D-Stereo-2x-384x704**](configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e_384704.py)                  | 37.3     | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-Occ-R50-4DLongterm-Stereo-2x-384x704**](configs/bevdet_occ/bevdet-occ-r50-4dlongterm-stereo-24e_384704.py)  | 39.3     | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |
| [**BEVDet-Occ-STBase-4D-Stereo-2x**](configs/bevdet_occ/bevdet-occ-stbase-4d-stereo-512x1408-24e.py) (*)                  | 42.0     | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) | [baidu](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) |

(*) Load 3D detection checkpoint.

## Get Started

#### Installation and Data Preparation

step 1. Please prepare environment as that in [Docker](docker/Dockerfile).

step 2. Clone the repo and install by.
```shell script
pip install -v -e .
```

step 3. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for BEVDet by running:
```shell
python tools/create_data_bevdet.py
```
step 4. For Occupancy Prediction task, download (only) the 'gts' from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and arrange the folder as:
```shell script
└── nuscenes
    ├── v1.0-trainval (existing)
    ├── sweeps  (existing)
    ├── samples (existing)
    └── gts (new)
```

#### Train model
```shell
# single gpu
python tools/train.py $config
# multiple gpu
./tools/dist_train.sh $config num_gpu
```

#### Test model
```shell
# single gpu
python tools/test.py $config $checkpoint --eval mAP
# multiple gpu
./tools/dist_test.sh $config $checkpoint num_gpu --eval mAP
```

#### Estimate the inference speed of BEVDet

```shell
# with pre-computation acceleration
python tools/analysis_tools/benchmark.py $config $checkpoint --fuse-conv-bn
# 4D with pre-computation acceleration
python tools/analysis_tools/benchmark_sequential.py $config $checkpoint --fuse-conv-bn
# view transformer only
python tools/analysis_tools/benchmark_view_transformer.py $config $checkpoint
```

#### Estimate the flops of BEVDet

```shell
python tools/analysis_tools/get_flops.py configs/bevdet/bevdet-r50.py --shape 256 704
```

#### Visualize the predicted result.

- Private implementation. (Visualization remotely/locally)

```shell
python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath
python tools/analysis_tools/vis.py $savepath/pts_bbox/results_nusc.json
```

#### Convert to TensorRT and test inference speed.

```shell
1. install mmdeploy from https://github.com/HuangJunJie2017/mmdeploy
2. convert to TensorRT
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16 --int8
3. test inference speed
python tools/analysis_tools/benchmark_trt.py $config $engine
```

## Acknowledgement


The finetuning code is based on [BEVDet](https://github.com/HuangJunJie2017/BEVDet). 