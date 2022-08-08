import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import InferDataset, collate_func
from pointpillars import PointPillar

from cfgs.kitti_cfg import cfg as cfg

from viewer import draw


def load_openpcdet_ckpt(model, path):
    from collections import OrderedDict
    ckpt = torch.load(path)['model_state']
    office_keys = list(ckpt.keys())[1:]
    my_keys = model.state_dict().keys()
    convert_model = OrderedDict()
    for k1, k2 in zip(my_keys, office_keys):
        convert_model[k1] = ckpt[k2]
        # print(k1, convert_model[k1].shape)
    model.load_state_dict(convert_model, strict=True)
    print(f'ckpt {path} load done!')
    return model


print(cfg)
ds = InferDataset(cfg)
dl = DataLoader(ds, batch_size=1, collate_fn=collate_func)

model = PointPillar(cfg)
if cfg.device == 'gpu':
    model = model.cuda()
tensor_key = ['voxel_feats', 'voxel_coords']

model = load_openpcdet_ckpt(model, 'openpcdet-office-ckpt/pointpillar_7728.pth')
model.eval()

print('start inference...')
result = {}
for data_dict in dl:
    if cfg.device == 'gpu':
        for k, v in data_dict.items():
            if k in tensor_key:
                data_dict[k] = v.cuda()

    data_dict = model(data_dict)

    for i in range(len(data_dict['cloud_name'])):
        name = os.path.basename(data_dict['cloud_name'][i])
        pred = data_dict['pred'][i]
        result[name] = pred
        result[name].update({'path': data_dict['cloud_name'][i]})
        print(f'point cloud {name} detect {len(pred["pred_labels"])} objects')

    break  # just demo so...

for name, pred in result.items():
    points = np.fromfile(pred['path'], dtype=np.float32).reshape(-1, 4)
    boxes = pred['pred_boxes'].cpu().numpy()
    draw(points, boxes)



