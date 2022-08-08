from easydict import EasyDict

cfg = EasyDict({
    # data cfg
    'cloud_path': './kitti_test_data',
    'point_dim': 4,
    'class_name': ['Car', 'Pedestrian', 'Cyclist'],
    'device': 'gpu',

    # voxelization cfg
    'pc_range': [0, -39.68, -3, 69.12, 39.68, 1],
    'voxel_size': [0.16, 0.16, 4],
    'use_abslote_xyz': True,
    'max_num_points': 64,

    # pillar vfe cfg
    'vfe_in_channels': 10,
    'num_bev_features': 64,

    # bev backbone cfg
    'layer_nums': [3, 5, 5],
    'layer_strides': [2, 2, 2],
    'num_filters': [64, 128, 256],
    'upsample_strides': [1, 2, 4],
    'num_upsample_filters': [128, 128, 128],

    # head cfg
    'input_channels': 384,
    'use_direction_classifier': True,
    'dir_offset': 0.78539,
    'dir_limit_offset': 0.0,
    'num_dir_bins': 2,
    'anchor_generator_cfg': [
        EasyDict({
            'class_name': 'Car',
            'anchor_sizes': [[3.9, 1.6, 1.56]],
            'anchor_rotations': [0, 1.57],
            'anchor_bottom_heights': [-1.78],
            'align_center': False,
            'feature_map_stride': 2,
            'matched_threshold': 0.6,
            'unmatched_threshold': 0.45
        }),
        EasyDict({
            'class_name': 'Pedestrian',
            'anchor_sizes': [[0.8, 0.6, 1.73]],
            'anchor_rotations': [0, 1.57],
            'anchor_bottom_heights': [-0.6],
            'align_center': False,
            'feature_map_stride': 2,
            'matched_threshold': 0.5,
            'unmatched_threshold': 0.35
        }),
        EasyDict({
            'class_name': 'Cyclist',
            'anchor_sizes': [[1.76, 0.6, 1.73]],
            'anchor_rotations': [0, 1.57],
            'anchor_bottom_heights': [-0.6],
            'align_center': False,
            'feature_map_stride': 2,
            'matched_threshold': 0.5,
            'unmatched_threshold': 0.35
        })
    ],

    # postprocess cfg
    'score_th': 0.1,
    'nms_cfg': EasyDict({
        'nms_th': 0.01,
        'nms_pre_maxsize': 4096,
        'nms_post_maxsize': 500
    })
})