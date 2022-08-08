import numpy as np
import torch
import cv2


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def naive_3diou(keep_box, res_boxes):
    '''
    keep_box: (7, ), x,y,z,dx,dy,dz,heading
    res_boxes: (n, 7)
    '''
    keep_box_min_x = keep_box[0] - keep_box[3] / 2  # scalar
    keep_box_max_x = keep_box[0] + keep_box[3] / 2
    res_boxes_min_x = res_boxes[:, 0] - res_boxes[:, 3] / 2  # (n, )
    res_boxes_max_x = res_boxes[:, 0] + res_boxes[:, 3] / 2
    min_x = np.maximum(res_boxes_min_x, keep_box_min_x)  # (n, )
    max_x = np.minimum(res_boxes_max_x, keep_box_max_x)  # (n, )
    x_overlap = max_x - min_x  # (n, )
    # y
    keep_box_min_y = keep_box[1] - keep_box[4] / 2
    keep_box_max_y = keep_box[1] + keep_box[4] / 2
    res_boxes_min_y = res_boxes[:, 1] - res_boxes[:, 4] / 2
    res_boxes_max_y = res_boxes[:, 1] + res_boxes[:, 4] / 2
    min_y = np.maximum(res_boxes_min_y, keep_box_min_y)
    max_y = np.minimum(res_boxes_max_y, keep_box_max_y)
    y_overlap = max_y - min_y
    # z
    keep_box_min_z = keep_box[2] - keep_box[5] / 2
    keep_box_max_z = keep_box[2] + keep_box[5] / 2
    res_boxes_min_z = res_boxes[:, 2] - res_boxes[:, 5] / 2
    res_boxes_max_z = res_boxes[:, 2] + res_boxes[:, 5] / 2
    min_z = np.maximum(res_boxes_min_z, keep_box_min_z)
    max_z = np.minimum(res_boxes_max_z, keep_box_max_z)
    z_overlap = max_z - min_z

    overlap_volumn = x_overlap * y_overlap * z_overlap  # (n, )
    keep_box_volumn = (keep_box_max_x - keep_box_min_x) * (keep_box_max_y - keep_box_min_y) * (
                keep_box_max_z - keep_box_min_z)  # scalar
    res_boxes_volumn = (res_boxes_max_x - res_boxes_min_x) * (res_boxes_max_y - res_boxes_min_y) * (
                res_boxes_max_z - res_boxes_min_z)  # (n, )
    total_volumn = (keep_box_volumn + res_boxes_volumn) - overlap_volumn  # (n, )
    ious = overlap_volumn / total_volumn  # (n, )
    return ious


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def draw_mask(corners, blank_map):
    '''
    corners: (8, 3), x,y,z
    blank_map: (H, W)
    '''
    # contours: (n,1,2)
    contours = corners[:4, :2]  # (4, 2), float
    contours = contours.astype(np.int32)
    H, W = blank_map.shape
    contours[:, 0] = np.clip(contours[:, 0], 0, W)
    contours[:, 1] = np.clip(contours[:, 1], 0, H)
    contours = np.expand_dims(contours, axis=1)  # (4, 1, 2), float
    img = blank_map.copy()

    cv2.drawContours(img, [contours], -1, 1, -1)
    return img.astype(np.bool)


def mask_3diou(keep_box, res_boxes, scale_factor=100):
    '''
    keep_box: (7, ), x,y,z,dx,dy,dz,heading
    res_boxes: (n, 7), x,y,z,dx,dy,dz,heading
    scale_factor: enlarge points coordinates to transform them into voxels
    returns:
    ious: (n, )
    '''
    keep_box = keep_box.copy()
    res_boxes = res_boxes.copy()
    keep_box[: 6] *= scale_factor
    res_boxes[:, :6] *= scale_factor
    all_boxes = np.concatenate((np.expand_dims(keep_box, axis=0), res_boxes), axis=0)  # (n+1, 7)
    all_corners = boxes_to_corners_3d(all_boxes)  # (n+1, 8, 3)
    all_min_x = np.min(all_corners[:, :, 0])
    all_max_x = np.max(all_corners[:, :, 0])
    all_min_y = np.min(all_corners[:, :, 1])
    all_max_y = np.max(all_corners[:, :, 1])
    H = int(all_max_y - all_min_y)
    W = int(all_max_x - all_min_x)

    all_corners[:, :, :2] -= [all_min_x, all_min_y]
    keep_corners = all_corners[0, :, :]  # (8, 3)
    res_corners = all_corners[1:, :, :]  # (n, 8, 3)

    blank_map = np.zeros((H, W))

    keep_mask = draw_mask(keep_corners, blank_map)
    # plt.imshow(keep_mask)
    # plt.show()

    # z
    keep_box_min_z = keep_box[2] - keep_box[5] / 2
    keep_box_max_z = keep_box[2] + keep_box[5] / 2
    res_boxes_min_z = res_boxes[:, 2] - res_boxes[:, 5] / 2
    res_boxes_max_z = res_boxes[:, 2] + res_boxes[:, 5] / 2
    min_z = np.maximum(res_boxes_min_z, keep_box_min_z)  # overlap area
    max_z = np.minimum(res_boxes_max_z, keep_box_max_z)
    z_overlap = max_z - min_z  # (n, )

    ious = np.zeros(len(z_overlap))  # (n, )
    for i in range(len(z_overlap)):
        h_i = z_overlap[i]
        mask_i = draw_mask(res_corners[i, :, :], blank_map)
        overlap_i = h_i * (np.sum(mask_i & keep_mask))
        union = np.sum(keep_mask) * keep_box[5] + np.sum(mask_i) * res_boxes[i, 5]
        ious[i] = overlap_i / union

    return ious


def nms_3d(boxes, scores, nms_thres=0.1, score_thres=0.6):
    '''
    boxes: (n, 7), x,y,z,dx,dy,dz,heading
    scores: (n, )
    returns:
    filtered_boxes: (n2, 7)
    keep_inds: indices of filtered boxes
    '''
    sorted_inds = np.argsort(-scores)
    keep_inds = []
    while len(sorted_inds)>0:
        if scores[sorted_inds[0]] < score_thres:
            break
        keep_inds.append(sorted_inds[0])
        if len(sorted_inds) == 1:
            break
        keep_box = boxes[sorted_inds[0], :]
        res_inds = sorted_inds[1:]
        res_boxes = boxes[res_inds, :]
        se_inds = np.arange(len(res_inds))
        # x
        keep_box_min_x = keep_box[0] - keep_box[3]/2  # scalar
        keep_box_max_x = keep_box[0] + keep_box[3]/2
        res_boxes_min_x = res_boxes[:, 0] - res_boxes[:, 3]/2 # (n, )
        res_boxes_max_x = res_boxes[:, 0] + res_boxes[:, 3]/2
        min_x = np.maximum(res_boxes_min_x, keep_box_min_x) # (n, )
        max_x = np.minimum(res_boxes_max_x, keep_box_max_x) # (n, )
        x_overlap = max_x > min_x
        # y
        keep_box_min_y = keep_box[1] - keep_box[4]/2
        keep_box_max_y = keep_box[1] + keep_box[4]/2
        res_boxes_min_y = res_boxes[:, 1] - res_boxes[:, 4]/2
        res_boxes_max_y = res_boxes[:, 1] + res_boxes[:, 4]/2
        min_y = np.maximum(res_boxes_min_y, keep_box_min_y)
        max_y = np.minimum(res_boxes_max_y, keep_box_max_y)
        y_overlap = max_y > min_y
        # z
        keep_box_min_z = keep_box[2] - keep_box[5]/2
        keep_box_max_z = keep_box[2] + keep_box[5]/2
        res_boxes_min_z = res_boxes[:, 2] - res_boxes[:, 5]/2
        res_boxes_max_z = res_boxes[:, 2] + res_boxes[:, 5]/2
        min_z = np.maximum(res_boxes_min_z, keep_box_min_z)
        max_z = np.minimum(res_boxes_max_z, keep_box_max_z)
        z_overlap = max_z > min_z

        overlap_mask = x_overlap & y_overlap & z_overlap
        care_res_boxes = res_boxes[overlap_mask, :] # (m, 7)
        overlap_se_inds = se_inds[overlap_mask]
        ious = mask_3diou(keep_box, care_res_boxes)  # (m, )
        delete_mask = ious > nms_thres
        delete_se_inds = overlap_se_inds[delete_mask]
        sorted_inds = np.delete(res_inds, delete_se_inds)

    if len(keep_inds)>0:
        keep_inds = np.array(keep_inds)
        return scores[keep_inds], keep_inds
    else:
        return None, None
