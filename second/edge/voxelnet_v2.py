# -*- coding: utf-8 -*-
# Filename : voxelnet_v2
__author__ = 'Xumiao Zhang'


import time
from enum import Enum
from functools import reduce

import numpy as np
import sparseconvnet as scn
import torch
from torch import nn
from torch.nn import functional as F

import torchplus
from torchplus import metrics
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
                                          WeightedSmoothL1LocalizationLoss,
                                          WeightedSoftmaxClassificationLoss)
from second.pytorch.models.pointpillars import PillarFeatureNet, PointPillarsScatter
from second.pytorch.utils import get_paddings_indicator

from second.protos import second_pb2
from second.pytorch.builder import losses_builder
from second.pytorch.models.voxelnet import LossNormType

from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.builder import (box_coder_builder)

class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 name='rpn'):
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        if use_bev:
            self.bev_extractor = Sequential(
                Conv2d(6, 32, 3, padding=1),
                BatchNorm2d(32),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),
                Conv2d(32, 64, 3, padding=1),
                BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_filters, num_filters[0], 3, stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

    def forward(self, x, bev=None):
        x = self.block1(x)
        up1 = self.deconv1(x)
        if self._use_bev:
            bev[:, -1] = torch.clamp(
                torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
            x = torch.cat([x, self.bev_extractor(bev)], dim=1)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="SparseMiddleExtractor",
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_sparse_rpn=False,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 use_bev=False,
                 lidar_only=False,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='voxelnet',
                 voxel_feature=False,
                 spatial_feature=False,
                 direct_predict=False):
        super().__init__()
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_sparse_rpn = use_sparse_rpn
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self._lidar_only = lidar_only
        self.target_assigner = target_assigner
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()

        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight

        self.voxel_feature = voxel_feature  # xumiao, generate voxel_feature?
        self.spatial_feature = spatial_feature  # xumiao, generate spatial_feature?
        self.direct_predict = direct_predict  # xumiao, predict with voxel_feature/spatial_feature?

        # print(pc_range)  # xumiao
        self.voxel_feature_extractor = PillarFeatureNet(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance,
            voxel_size=voxel_size,
            pc_range=pc_range
        )

        self.middle_feature_extractor = PointPillarsScatter(output_shape=output_shape, num_input_features=vfe_num_filters[-1])
        num_rpn_input_filters = self.middle_feature_extractor.nchannels

        rpn_class_dict = {
            "RPN": RPN,
        }
        rpn_class = rpn_class_dict[rpn_class_name]
        self.rpn = rpn_class(
            use_norm=True,
            num_class=num_class,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_filters=num_rpn_input_filters,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_bev=use_bev,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=target_assigner.box_coder.code_size)

        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def forward(self, example, voxel_features=[]):
        """module's forward should always accept dict and return loss.
        """
        if not self.direct_predict:
            voxels = example["voxels"]
            num_points = example["num_points"]
            coors = example["coordinates"]
            # if len(num_points.shape) == 2:  # multi-gpu, xumiao
            #     num_voxel_per_batch = example["num_voxels"].cpu().numpy().reshape(
            #         -1)
            #     voxel_list = []
            #     num_points_list = []
            #     coors_list = []
            #     for i, num_voxel in enumerate(num_voxel_per_batch):
            #         voxel_list.append(voxels[i, :num_voxel])
            #         num_points_list.append(num_points[i, :num_voxel])
            #         coors_list.append(coors[i, :num_voxel])
            #     voxels = torch.cat(voxel_list, dim=0)
            #     num_points = torch.cat(num_points_list, dim=0)
            #     coors = torch.cat(coors_list, dim=0)
            batch_anchors = example["anchors"]
            batch_size_dev = batch_anchors.shape[0]
            t = time.time()
            # features: [num_voxels, max_num_points_per_voxel, 7]
            # num_points: [num_voxels]
            # coors: [num_voxels, 4]
        
            voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
            if self.voxel_feature:
                return voxel_features  # num_voxels * 64

            
            spatial_features = self.middle_feature_extractor(voxel_features, coors, batch_size_dev)
            if self.spatial_feature:
                return spatial_features  # 1 * 64 * 496 * 432 (grid size: 496*432)

        if self.voxel_feature:  # self.direct_predict == True, self.voxel_feature == True
            voxel_features=torch.tensor(voxel_features,  device='cuda:0')
            coors = example["coordinates"]
            batch_anchors = example["anchors"]
            batch_size_dev = batch_anchors.shape[0]
            spatial_features = self.middle_feature_extractor(voxel_features, coors, batch_size_dev)
        #  if self.spatial_feature:
            #  spatial_features = example (to modify)
        
        preds_dict = self.rpn(spatial_features)
        # self._total_forward_time += time.time() - t
        
        return self.predict(example, preds_dict)

    def predict(self, example, preds_dict):
        t = time.time()
        batch_size = example['anchors'].shape[0]
        batch_anchors = example["anchors"].view(batch_size, -1, 7)

        self._total_inference_count += batch_size

        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
        batch_imgidx = example['image_idx']

        self._total_forward_time += time.time() - t
        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []

        for box_preds, cls_preds, dir_preds, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds,
                batch_imgidx, batch_anchors_mask
        ):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                # print(dir_preds.shape)
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            if self._use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms
            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            if self._multiclass_nms:
                # curently only support class-agnostic boxes.
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
                selected_per_class = box_torch_ops.multiclass_nms(
                    nms_func=nms_func,
                    boxes=boxes_for_mcnms,
                    scores=total_scores,
                    num_class=self._num_class,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                    score_thresh=self._nms_score_threshold,
                )
                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []
                for i, selected in enumerate(selected_per_class):
                    if selected is not None:
                        num_dets = selected.shape[0]
                        selected_boxes.append(box_preds[selected])
                        selected_labels.append(
                            torch.full([num_dets], i, dtype=torch.int64))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(dir_labels[selected])
                        selected_scores.append(total_scores[selected, i])
                if len(selected_boxes) > 0:
                    selected_boxes = torch.cat(selected_boxes, dim=0)
                    selected_labels = torch.cat(selected_labels, dim=0)
                    selected_scores = torch.cat(selected_scores, dim=0)
                    if self._use_direction_classifier:
                        selected_dir_labels = torch.cat(
                            selected_dir_labels, dim=0)
                else:
                    selected_boxes = None
                    selected_labels = None
                    selected_scores = None
                    selected_dir_labels = None
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)
                else:
                    top_scores, top_labels = torch.max(total_scores, dim=-1)

                if self._nms_score_threshold > 0.0:
                    thresh = torch.tensor(
                        [self._nms_score_threshold],
                        device=total_scores.device).type_as(total_scores)
                    top_scores_keep = (top_scores >= thresh)
                    top_scores = top_scores.masked_select(top_scores_keep)
                if top_scores.shape[0] != 0:
                    if self._nms_score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                        iou_threshold=self._nms_iou_threshold,
                    )
                else:
                    selected = None
                if selected is not None:
                    selected_boxes = box_preds[selected]
                    if self._use_direction_classifier:
                        selected_dir_labels = dir_labels[selected]
                    selected_labels = top_labels[selected]
                    selected_scores = top_scores[selected]
            # finally generate predictions.

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.bool()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
                    # box_preds[..., -1] += (
                    #     ~(dir_labels.byte())).type_as(box_preds) * np.pi
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                # predictions
                predictions_dict = {
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": img_idx,
                }
            else:
                predictions_dict = {
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None,
                    "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)
        self._total_postprocess_time += time.time() - t
        return predictions_dicts


def build(model_cfg: second_pb2.VoxelNet, voxel_generator, target_assigner, voxel_feature=False, spatial_feature=False, direct_predict=False) -> VoxelNet:
    """build second pytorch instance.
    """
    if not isinstance(model_cfg, second_pb2.VoxelNet):
        raise ValueError('model_cfg not of type ' 'second_pb2.VoxelNet.')
    vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)
    vfe_with_distance = model_cfg.voxel_feature_extractor.with_distance
    grid_size = voxel_generator.grid_size
    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    num_class = model_cfg.num_class

    num_input_features = model_cfg.num_point_features
    if model_cfg.without_reflectivity:
        num_input_features = 3
    loss_norm_type_dict = {
        0: LossNormType.NormByNumExamples,
        1: LossNormType.NormByNumPositives,
        2: LossNormType.NormByNumPosNeg,
    }
    loss_norm_type = loss_norm_type_dict[model_cfg.loss_norm_type]

    losses = losses_builder.build(model_cfg.loss)
    encode_rad_error_by_sin = model_cfg.encode_rad_error_by_sin
    cls_loss_ftor, loc_loss_ftor, cls_weight, loc_weight, _ = losses
    pos_cls_weight = model_cfg.pos_class_weight
    neg_cls_weight = model_cfg.neg_class_weight
    direction_loss_weight = model_cfg.direction_loss_weight

    net = VoxelNet(
        dense_shape,
        num_class=num_class,
        vfe_class_name=model_cfg.voxel_feature_extractor.module_class_name,
        vfe_num_filters=vfe_num_filters,
        middle_class_name=model_cfg.middle_feature_extractor.module_class_name,
        middle_num_filters_d1=list(
            model_cfg.middle_feature_extractor.num_filters_down1),
        middle_num_filters_d2=list(
            model_cfg.middle_feature_extractor.num_filters_down2),
        rpn_class_name=model_cfg.rpn.module_class_name,
        rpn_layer_nums=list(model_cfg.rpn.layer_nums),
        rpn_layer_strides=list(model_cfg.rpn.layer_strides),
        rpn_num_filters=list(model_cfg.rpn.num_filters),
        rpn_upsample_strides=list(model_cfg.rpn.upsample_strides),
        rpn_num_upsample_filters=list(model_cfg.rpn.num_upsample_filters),
        use_norm=True,
        use_rotate_nms=model_cfg.use_rotate_nms,
        multiclass_nms=model_cfg.use_multi_class_nms,
        nms_score_threshold=model_cfg.nms_score_threshold,
        nms_pre_max_size=model_cfg.nms_pre_max_size,
        nms_post_max_size=model_cfg.nms_post_max_size,
        nms_iou_threshold=model_cfg.nms_iou_threshold,
        use_sigmoid_score=model_cfg.use_sigmoid_score,
        encode_background_as_zeros=model_cfg.encode_background_as_zeros,
        use_direction_classifier=model_cfg.use_direction_classifier,
        use_bev=model_cfg.use_bev,
        num_input_features=num_input_features,
        num_groups=model_cfg.rpn.num_groups,
        use_groupnorm=model_cfg.rpn.use_groupnorm,
        with_distance=vfe_with_distance,
        cls_loss_weight=cls_weight,
        loc_loss_weight=loc_weight,
        pos_cls_weight=pos_cls_weight,
        neg_cls_weight=neg_cls_weight,
        direction_loss_weight=direction_loss_weight,
        loss_norm_type=loss_norm_type,
        encode_rad_error_by_sin=encode_rad_error_by_sin,
        loc_loss_ftor=loc_loss_ftor,
        cls_loss_ftor=cls_loss_ftor,
        target_assigner=target_assigner,
        voxel_size=voxel_generator.voxel_size,
        pc_range=voxel_generator.point_cloud_range,
        voxel_feature=voxel_feature,
        spatial_feature=spatial_feature,
        direct_predict=direct_predict
    )
    return net


def build_network(model_cfg, voxel_feature=False, spatial_feature=False, direct_predict=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
    ######################
    # BUILD NET
    ######################
    net = build(model_cfg, voxel_generator, target_assigner, voxel_feature, spatial_feature, direct_predict)
    net.cuda()
    return net,voxel_generator


# self.voxel_feature_extractor = PillarFeatureNet(
#                 num_input_features,
#                 use_norm,
#                 num_filters=vfe_num_filters,
#                 with_distance=with_distance,
#                 voxel_size=voxel_size,
#                 pc_range=pc_range)


# # build function
# num_input_features = model_cfg.num_point_features  # 4
# use_norm = True
# vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)  # [64]
# vfe_with_distance = model_cfg.voxel_feature_extractor.with_distance  # false
# voxel_size=voxel_generator.voxel_size  # voxel_size : [0.16, 0.16, 4]     
# pc_range=voxel_generator.point_cloud_range  # [0, -39.68, -3, 69.12, 39.68, 1]
