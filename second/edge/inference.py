# -*- coding: utf-8 -*-
# Filename : inference
__author__ = 'Xumiao Zhang'


import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# sys.path.append("..")
# sys.path.append("../..")
sys.path.append("/src/main/cpp/pointpillars/second.pytorch")
sys.path.append("/src/main/cpp/pointpillars/second.pytorch/second/")
sys.path.append("/src/main/cpp/pointpillars/second.pytorch/SparseConvNet")
import numpy as np
import torchplus
import torch
from google.protobuf import text_format
# from second.utils import simplevis
from voxelnet_v2 import build_network  # from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.core import box_np_ops
from collections import OrderedDict

config_path = "/src/main/cpp/pointpillars/second.pytorch/second/configs/pointpillars/car/kitti_gta_v2.proto"  # xyres_16 gta_0403 gta_0408
ckpt_path = "/src/main/cpp/pointpillars/second.pytorch/second/models/kitti_gta_v2"  #xumiao jiachen kitti_gta gta_0406 gta_0408 
data_path = '/home/xumiao/Edge/test/car/'  # data_path = '/home/xumiao/Edge/reduced/' 0407/
save_path = '/home/xumiao/Edge/test/car/'  # save_path = '/home/xumiao/Edge/inference/' 0407/


def example_convert_to_torch(example, dtype=torch.float32, device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2"
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v
    return example_torch

def kitti_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_z', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key in ['rotation_z', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key in ['dimensions', 'location']:
            if val is None:
                print(key)
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
    return ' '.join(res_line)

class Inference():
    def __init__(self):
        self.config = None
        self.model_cfg = None
        self.center_limit_range = None
        self.out_size_factor = None
        self.max_voxels = None
        self.voxel_size = None
        self.pc_range = None
        self.grid_size = None

        self.anchors = None
        self.anchors_bv = None

        self.net = None
        self.voxel_generator = None
        

    def read_config(self):
        self.config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, self.config)
        self.model_cfg = self.config.model.second
        self.center_limit_range = self.model_cfg.post_center_limit_range
        self.out_size_factor = self.model_cfg.rpn.layer_strides[0] // self.model_cfg.rpn.upsample_strides[0]
        self.max_voxels = self.config.eval_input_reader.max_number_of_voxels
    
    def build_model(self):
        ### model
        self.net, self.voxel_generator = build_network(self.model_cfg)
        self.net.eval()
        torchplus.train.try_restore_latest_checkpoints(ckpt_path, [self.net])
        target_assigner = self.net.target_assigner
        self.voxel_size = self.voxel_generator.voxel_size
        self.pc_range = self.voxel_generator.point_cloud_range
        # print(self.pc_range)
        self.grid_size = self.voxel_generator.grid_size
        
        ### other parameters
        feature_map_size = self.grid_size[:2] // self.out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        self.anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
        self.anchors = anchors.reshape([1,-1, 7])
    
    def execute_model(self, points):
        points = points.reshape(-1,4)
        ### point cloud
        voxels, coordinates, num_points = self.voxel_generator.generate(points, self.max_voxels)
        # print('coordinates', np.sum(coordinates), coordinates)
    
        ### other parameters (cont'd)
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(coordinates, tuple(self.grid_size[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_np_ops.fused_get_anchors_area(dense_voxel_map, self.anchors_bv, self.voxel_size, self.pc_range, self.grid_size)
        anchors_mask = anchors_area > 1
        coordinates = np.pad(coordinates, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        anchors_mask = anchors_mask.reshape([1,-1])
    
        ### organize example
        example = {
            'voxels': voxels,  # n*100*4, voxels
            'num_points': num_points,  # n*1, number of points per voxel
            'coordinates': coordinates,  # n*4
            "num_voxels": np.array([voxels.shape[0]], dtype=np.int64),  # number of voxels, n
            "anchors": self.anchors,  # 1*m*7
            'anchors_mask': anchors_mask,  # 1*m, boolean, all False
            'image_idx': np.array([[0]])
        }
        # print('total_points', points.shape[0])
        # print('num_voxels', voxels.shape[0])
        example = example_convert_to_torch(example)
    
        ### predict
        with torch.no_grad():
            pred = self.net(example)[0]
    
        ### save results
        scores = pred["scores"].data.cpu().numpy()
        box_preds_lidar = pred["box3d_lidar"].data.cpu().numpy()
        box_preds_lidar = box_preds_lidar[:, [0, 1, 2, 5, 3, 4, 6]]  # xyzwlh->xyzhwl(label file format)
        label_preds = pred["label_preds"].data.cpu().numpy()
        result_lines = []
        for box_lidar, score in zip(box_preds_lidar, scores):
            limit_range = self.center_limit_range  # np.array([0, -39.68, -5, 69.12, 39.68, 5])
            if (np.any(box_lidar[:3] < limit_range[:3]) or np.any(box_lidar[:3] > limit_range[3:])):
                continue
            result_dict = {
                # 'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box_lidar[6],
                'location': box_lidar[:3],
                'dimensions': box_lidar[3:6],
                'rotation_z': box_lidar[6],
                'score': score,
            }
            result_line = kitti_result_line(result_dict)
            result_lines.append(result_line)
        result_str = '\n'.join(result_lines)
        return result_str
    

if __name__ == '__main__':
    
    # inf = Inference()
    # inf.read_config()
    # inf.build_model()
    # for file in os.listdir(data_path):
    #     if file.split('.')[1] == 'bin' and 'crop' in file.split('.')[0] and not 'box' in file.split('.')[0]:
    #         print(data_path+file)
    #         points = np.fromfile(data_path+file, dtype=np.float32, count=-1).reshape([-1, 4])    
    #         result_file = save_path + file.split('.')[0] + '.txt'
    #         result_str = inf.execute_model(points)

    #         with open(result_file, 'w') as f:
    #             f.write(result_str)



    # points = np.fromfile('/home/xumiao/DeepGTAV-data/gta_0601/training/velodyne/000050.bin', dtype=np.float32, count=-1).reshape([-1, 4])    
    # np.savetxt('/home/xumiao/sample1.txt', points)
    # result_str = inf.execute_model(points)

    # with open('/home/xumiao/sample2.txt', 'w') as f:
    #     f.write(result_str)



    data_file = data_path + sys.argv[1] + '.bin'
    # result_file = save_path + sys.argv[1] + '.txt'
    points = np.fromfile(data_file, dtype=np.float32, count=-1).reshape([-1, 4])

    inf = Inference()
    inf.read_config()
    inf.build_model()
    result_str = inf.execute_model(points)

    # with open(result_file, 'w') as f:
    #     f.write(result_str)



    # inf = Inference()
    # inf.read_config()
    # inf.build_model()
    # for name in ['000039.bin', '000042.bin', '000046.bin']:
    #     points = np.fromfile('/home/xumiao/Edge/test/'+name, dtype=np.float32, count=-1).reshape([-1, 4])
    #     t1 = time.time()
    #     result_str = inf.execute_model(points)
    #     t2 = time.time()
    #     print(t2-t1)
