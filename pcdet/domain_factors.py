import os, sys
#sys.path.append('/data/wy/workspace/ST3D/pcdet/ops/roiaware_pool3d')
from ops.roiaware_pool3d import roiaware_pool3d_utils
import numpy as np
import multiprocessing
import torch
import pickle
import open3d as o3d

data_path = '/data/wy/kitti/training/original_velodyne/'
save_path = '/data/wy/kitti/training/samples/'
os.makedirs(os.path.join(save_path, 'object_size'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'weather'), exist_ok=True)

with open('/data/wy/workspace/ST3D/data/kitti/kitti_infos_trainval.pkl', 'rb') as f:
    infos = pickle.load(f)

def process(pc_idx):
    pcd = o3d.geometry.PointCloud()

    print(pc_idx)
    bin_path = data_path + '{:06d}.bin'.format(pc_idx)
    raw_data = np.fromfile(bin_path, dtype=np.float32)
    points = raw_data.reshape(-1, 4).astype(np.float32)

    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud('ply/{}.ply'.format(pc_idx), pcd)

    info = infos[pc_idx]
    annos = info['annos']
    names = annos['name']
    gt_boxes = annos['gt_boxes_lidar']

    num_obj = gt_boxes.shape[0]
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)
    
    for i in range(num_obj):
        if names[i] != 'Car':
            continue
        gt_points = points[point_indices[i] > 0]

        gt_points[:, :3] -= gt_boxes[i, :3]
        gt_points[:, :3] *= 1.2


    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud('ply/{}_size.ply'.format(pc_idx), pcd)

    points.tofile(os.path.join(save_path, 'object_size', '{:06d}.bin'.format(pc_idx)))

process(3)
'''
p = multiprocessing.Pool(16)
info_list = list(range(7481))
p.map_async(process, info_list)
p.close()
p.join()
'''