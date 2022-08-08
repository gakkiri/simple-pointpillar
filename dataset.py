import torch
from collections import defaultdict
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from utils import mask_points_by_range


def collate_func(batch_list):
    voxel_feat_list = []
    voxel_coord_list = []
    voxel_num_points_list = []
    name_list = []
    for i, batch in enumerate(batch_list):
        voxel_feat_list.append(torch.tensor(batch['voxel_feats']))
        voxel_num_points_list.append(torch.tensor(batch['voxel_num_points']))
        name_list.append(batch['cloud_name'])

        coords = batch['voxel_coords']  # x, y
        coords = np.hstack([coords, np.ones((len(coords), 1)) * i])  # x, y, b
        voxel_coord_list.append(torch.tensor(coords))

    voxel_feat_list = torch.cat(voxel_feat_list).float()
    voxel_coord_list = torch.cat(voxel_coord_list).long()
    voxel_num_points_list = torch.cat(voxel_num_points_list).long()
    return {
        'voxel_feats': voxel_feat_list,
        'voxel_coords': voxel_coord_list,
        'voxel_num_points': voxel_num_points_list,
        'batch_size': len(batch_list),
        'cloud_name': name_list
    }


class InferDataset(Dataset):
    def __init__(self, cfg):
        self.cloud_path = glob(cfg.cloud_path + '/*.bin')

        self.cloud_path.sort()

        self.pc_range = cfg.pc_range
        self.voxel_size = cfg.voxel_size
        self.max_num_points = cfg.max_num_points
        self.point_dim = cfg.point_dim
        self.use_abslote_xyz = cfg.use_abslote_xyz

        voxel_x_range = np.linspace(
            self.pc_range[0], self.pc_range[3],
            int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0] + 1)
        )
        voxel_y_range = np.linspace(
            self.pc_range[1], self.pc_range[4],
            int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1] + 1)
        )

        grid_X, grid_Y = np.meshgrid(voxel_x_range, voxel_y_range)
        self.flat_grid_X = grid_X.reshape(-1).round(5)
        self.flat_grid_Y = grid_Y.reshape(-1).round(5)

    def __getitem__(self, idx):
        raw_cloud = np.fromfile(self.cloud_path[idx], dtype=np.float32).reshape(-1, 4)[:, :self.point_dim]
        cloud = raw_cloud[mask_points_by_range(raw_cloud, self.pc_range)]

        voxel_dict = defaultdict(list)
        for pt in cloud:
            x, y, z = pt[:3]
            voxel_x = round(int(x / self.voxel_size[0]) * self.voxel_size[0], 5)
            voxel_y = round(int(y / self.voxel_size[1]) * self.voxel_size[1], 5)
            if len(voxel_dict[(voxel_y, voxel_x)]) >= self.max_num_points:
                continue
            voxel_dict[(voxel_y, voxel_x)].append(pt)

        voxel_feats = []
        voxel_coords = []
        num_points = []
        for voxel_corner, pts in voxel_dict.items():
            pts = np.array(pts)  # x, y, z, (r)
            if len(pts) < self.max_num_points:
                padding = np.zeros((self.max_num_points - len(pts), pts.shape[-1]))
                feats = np.concatenate([pts, padding])  # [64, point_dim]
            else:
                feats = pts

            cluster = pts[:, :3].mean(0).reshape(1, -1).repeat(len(feats), axis=0)  # cx, cy, cz
            feat_cluster = feats[:, :3] - cluster  # cx, cy, cz

            voxel_ct_y = voxel_corner[0] + self.voxel_size[0] / 2
            voxel_ct_x = voxel_corner[1] + self.voxel_size[1] / 2
            voxel_ct_z = (self.pc_range[5] + self.pc_range[2]) / 2
            voxel_ct = np.array([voxel_ct_x, voxel_ct_y, voxel_ct_z]).round(5)
            feat_center = feats[:, :voxel_ct.shape[-1]] - voxel_ct  # px, py, pz

            if self.use_abslote_xyz:
                feats = np.hstack([feats[:, :self.point_dim], feat_cluster, feat_center])  # x, y, z, cx, cy, cz, px, py, pz
            else:
                feats = np.hstack([feats[:, 3:], feat_cluster, feat_center])
            assert feats.shape == (self.max_num_points, 6 + self.point_dim), feats.shape

            feats[len(pts):, ...] = 0
            num_points.append(len(pts))
            voxel_feats.append(feats.reshape(1, self.max_num_points, -1))
            voxel_coords.append(np.array([
                (voxel_corner[1] - self.pc_range[0]) / self.voxel_size[0],
                (voxel_corner[0] - self.pc_range[1]) / self.voxel_size[1]
            ]))  # x, y

        voxel_feats = np.vstack(voxel_feats)
        voxel_coords = np.array(voxel_coords)
        num_points = np.array(num_points)
        return {
            'voxel_feats': voxel_feats,
            'voxel_coords': voxel_coords,
            'voxel_num_points': num_points,
            'cloud_name': self.cloud_path[idx]
        }

    def __len__(self):
        return len(self.cloud_path)