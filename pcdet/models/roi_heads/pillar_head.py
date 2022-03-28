import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, loss_utils


class PillarHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg, is_loss=False)
        self.model_cfg = model_cfg



    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                spatial_features_2d: (B, C, H, W)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'].detach()
        spatial_features_2d = batch_dict['spatial_features_2d'].detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)

        dataset_cfg = batch_dict['dataset_cfg']
        min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
        min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
        voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
        voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
        down_sample_ratio = self.model_cfg.ROI_GRID_POOL.DOWNSAMPLE_RATIO

        pooled_features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            # Map global boxes coordinates to feature map coordinates
            x1 = (rois[b_id, :, 0] - rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            x2 = (rois[b_id, :, 0] + rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            y1 = (rois[b_id, :, 1] - rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            y2 = (rois[b_id, :, 1] + rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)

            angle, _ = common_utils.check_numpy_to_torch(rois[b_id, :, 6])

            cosa = torch.cos(angle)
            sina = torch.sin(angle)

            theta = torch.stack((
                (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
            ), dim=1).view(-1, 2, 3).float()

            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            grid = nn.functional.affine_grid(
                theta,
                torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size))
            )

            pooled_features = nn.functional.grid_sample(
                spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width),
                grid
            )

            pooled_features_list.append(pooled_features)

        torch.backends.cudnn.enabled = True
        pooled_features = torch.cat(pooled_features_list, dim=0)

        return pooled_features

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        if 'mimic' in batch_dict.keys() and not self.training:
            targets_dict = self.proposal_layer(
                batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN']
            )
            targets_ = self.assign_targets(batch_dict.copy())
            batch_dict['rois_mimic'] = targets_['rois'].clone()
            targets_dict = self.proposal_layer(
                batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TEST']
            )

        else:
            targets_dict = self.proposal_layer(
                batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )
            if self.training:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']


        return batch_dict

    