from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["BEVFusionGeoMIM"]


def patchify(imgs, encoder_stride):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = encoder_stride
    n, c, h, w = imgs.shape
    assert h % p == 0 and w % p == 0

    h1, w1 = h // p, w // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h1, p, w1, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h1 * w1, p**2 * c))
    return x

def unpatchify(x, encoder_stride):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = encoder_stride
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    c = int(x.shape[2] / p / p)

    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs


@FUSIONMODELS.register_module()
class BEVFusionGeoMIM(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        bev_feat_dim=256,
        **kwargs,
    ) -> None:
        super().__init__()
        self.depth_loss = 0.0
        self.fuser = None
        self.decoder = None
        self.heads = None

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        for p in self.encoders['lidar'].parameters():
            p.requires_grad = False

        self.prediction_head = nn.Conv2d(
            encoders["camera"]["vtransform"]['out_channels'],
            bev_feat_dim,
            kernel_size=1, stride=1
        )

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x, x_depth = self.encoders["camera"]["backbone"](
            x, camera2ego, camera_intrinsics, img_aug_matrix
        )

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()

        num_batch_imgs = int(BN / (B * N))
        
        outputs = []
        for i in range(num_batch_imgs):
            this_x = x[B * N * i : B * N * (i + 1), ...]
            this_x = this_x.view(B, N, C, H, W)

            this_depth = x_depth[B * N * i : B * N * (i + 1), ...]
            this_depth = this_depth.view(B, N, C, H, W)

            this_x = self.encoders["camera"]["vtransform"](
                this_x,
                this_depth,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
            outputs.append(self.prediction_head(this_x))
            if self.encoders["camera"]["vtransform"].ret_depth:
                self.depth_loss = self.depth_loss + self.encoders["camera"]["vtransform"].depth_loss / num_batch_imgs

        return outputs

    @torch.no_grad()
    @force_fp32()
    def extract_lidar_features(self, x):
        self.encoders["lidar"]['voxelize'].eval()
        self.encoders["lidar"]['backbone'].eval()

        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                cam_feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                with torch.no_grad():
                    lidar_feature = self.extract_lidar_features(points)
                    # normalize within grid
                    lidar_feature = patchify(lidar_feature, 18)
                    mean = lidar_feature.mean(dim=-1, keepdim=True)
                    var = lidar_feature.var(dim=-1, keepdim=True)
                    lidar_feature = (lidar_feature - mean) / (var + 1.e-6)**.5
                    lidar_feature = unpatchify(lidar_feature, 18)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

        outputs = {}

        all_loss = 0.0
        for item in cam_feature:
            loss = (item - lidar_feature.detach()) ** 2.
            if torch.isnan(loss).any():
                # skip this iter may be a better solution
                print('rec loss nan, set to 0')
                loss = 0.0 * item
            loss = loss.mean()
            all_loss = all_loss + loss
        all_loss = all_loss / len(cam_feature)
        outputs['rec_loss'] = all_loss

        if self.depth_loss:
            outputs['depth_loss'] = self.depth_loss
            self.depth_loss = 0.0

        return outputs
