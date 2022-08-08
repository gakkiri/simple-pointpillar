import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from anchor import AnchorGenerator, ResidualCoder
from utils import nms_3d

__all__ = ['PointPillar']


class PillarVFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.norm = torch.nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, batch_dict):
        inputs = batch_dict['voxel_feats']
        x = self.linear(inputs)
        # torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        # torch.backends.cudnn.enabled = True
        x = torch.nn.functional.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        batch_dict['pillar_features'] = x_max.squeeze()
        return batch_dict


class PointPillarScatter(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.pc_range = cfg.pc_range
        self.voxel_size = cfg.voxel_size
        self.nx = int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0])
        self.ny = int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1])
        self.num_bev_features = cfg.num_bev_features

    def forward(self, batch_dict):
        pillar_feats, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_size = coords[:, -1].max().int().item() + 1

        batch_spatial_features = []
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nx * self.ny,
                dtype=pillar_feats.dtype,
                device=pillar_feats.device
            )

            batch_mask = coords[:, -1] == batch_idx
            this_coords = coords[batch_mask, :]  # x, y, b
            indices = this_coords[:, 1] * self.nx + this_coords[:, 0]
            indices = torch.tensor(indices).long()
            pillars = pillar_feats[batch_mask, :]
            spatial_feature[:, indices] = pillars.t()
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class BevBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        layer_nums = cfg.layer_nums
        layer_strides = cfg.layer_strides
        num_filters = cfg.num_filters
        upsample_strides = cfg.upsample_strides
        num_upsample_filters = cfg.num_upsample_filters
        input_channels = cfg.num_bev_features

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict


class AnchorHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device_ = cfg.device
        self.class_name = cfg.class_name
        self.num_class = len(self.class_name)
        self.pc_range = np.array(cfg.pc_range)
        self.voxel_size = np.array(cfg.voxel_size)
        self.grid_size = (self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size
        self.dir_offset = cfg.dir_offset
        self.dir_limit_offset = cfg.dir_limit_offset
        self.num_dir_bins = cfg.num_dir_bins

        self.box_coder = ResidualCoder()

        anchor_generator_cfg = cfg.anchor_generator_cfg
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=self.grid_size, point_cloud_range=self.pc_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = anchors
        if self.device_ == 'gpu':
            self.anchors = [x.cuda() for x in anchors]

        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        input_channels = cfg.input_channels
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        self.conv_dir_cls = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * cfg.num_dir_bins,
            kernel_size=1
        )

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        if isinstance(self.anchors, list):
            anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        dir_offset = self.dir_offset
        dir_limit_offset = self.dir_limit_offset
        dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
            else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
        dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

        period = (2 * np.pi / self.num_dir_bins)
        val = batch_box_preds[..., 6] - dir_offset
        dir_rot = val - torch.floor(val / period + dir_limit_offset) * period

        batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        return batch_cls_preds, batch_box_preds

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=data_dict['batch_size'],
            cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
        )
        data_dict['batch_cls_preds'] = batch_cls_preds
        data_dict['batch_box_preds'] = batch_box_preds
        data_dict['cls_preds_normalized'] = False
        return data_dict


class PointPillar(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.vfe = PillarVFE(cfg.vfe_in_channels, cfg.num_bev_features)
        self.scatter = PointPillarScatter(cfg)
        self.bev_backbone = BevBackbone(cfg)
        self.head = AnchorHead(cfg)

        self.num_class = len(cfg.class_name)
        self.score_th = cfg.score_th
        self.nms_cfg = cfg.nms_cfg

    @torch.no_grad()
    def post_process(self, data_dict):
        batch_size = data_dict['batch_size']

        pred_dict = []
        for batch_mask in range(batch_size):
            box_preds = data_dict['batch_box_preds'][batch_mask]
            cls_preds = data_dict['batch_cls_preds'][batch_mask]
            assert cls_preds.shape[1] in [1, self.num_class]

            if not data_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)

            cls_preds, label_preds = torch.max(cls_preds, dim=-1)
            label_preds += 1

            selected_scores, selected = nms_3d(
                box_preds.detach().cpu().numpy(), cls_preds.detach().cpu().numpy(),
                nms_thres=self.nms_cfg.nms_th, score_thres=self.score_th
            )
            final_labels = label_preds[selected]
            final_scores = selected_scores
            final_boxes = box_preds[selected]

            pred_dict.append({
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            })

        data_dict['pred'] = pred_dict
        return data_dict

    def forward(self, data_dict):
        data_dict = self.vfe(data_dict)
        data_dict = self.scatter(data_dict)
        data_dict = self.bev_backbone(data_dict)
        data_dict = self.head(data_dict)
        data_dict = self.post_process(data_dict)
        return data_dict
