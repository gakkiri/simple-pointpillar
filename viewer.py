import numpy as np
import open3d


box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])

        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        if 0:
            corners = box3d.get_box_points()
            vis.add_3d_label(corners[5], str(i))
    return vis


def draw(points, gt_boxes=None):
    points = points[:, :3]

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points)

    vis.add_geometry(pts)

    # vis.get_render_option().background_color = np.zeros(3)
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    batch = np.load('before_rotation.npy', allow_pickle=True).tolist()
    points = batch['points']
    gt_boxes = batch['gt_boxes']
    draw(points, gt_boxes)

