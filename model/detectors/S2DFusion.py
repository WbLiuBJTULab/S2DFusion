from .detector3d_template import Detector3DTemplate
from ..backbones_3d import vfe
from ...utils import common_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseObjectMapHead(nn.Module):
    def __init__(self, channels, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class LightImageEncoder(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, images):
        return self.net(images)


class DepthwiseInteraction(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3):
        super().__init__()
        layers = []
        cur_channels = in_channels
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(cur_channels, cur_channels, kernel_size=3, padding=1, groups=cur_channels, bias=False),
                nn.BatchNorm2d(cur_channels),
                nn.ReLU(),
                nn.Conv2d(cur_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])
            cur_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class S2DFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        obj_cfg = self.model_cfg.get('BEV_OBJECT_MAP', {})
        self.use_bev_object_map = obj_cfg.get('ENABLED', False)
        self.bev_object_map_weight = obj_cfg.get('LOSS_WEIGHT', 0.5)
        self.bev_gaussian_alpha = obj_cfg.get('GAUSSIAN_ALPHA', 0.3)
        self.bev_object_map_classes = obj_cfg.get('NUM_CLASSES', self.num_class)
        self.use_bev_object_fusion = obj_cfg.get('FUSION_ENABLED', True)
        if self.use_bev_object_map:
            self.bev_object_head = DepthwiseObjectMapHead(
                self.backbone_2d.num_bev_features, out_channels=self.bev_object_map_classes
            )
            if self.use_bev_object_fusion:
                self.bev_object_interaction = DepthwiseInteraction(
                    self.backbone_2d.num_bev_features + self.bev_object_map_classes,
                    self.backbone_2d.num_bev_features,
                    num_layers=obj_cfg.get('FUSION_LAYERS', 3)
                )
                self.bev_object_fusion_scale = nn.Parameter(torch.tensor(0.1))
        teacher_cfg = self.model_cfg.get('IMAGE_TEACHER', {})
        self.use_image_teacher = teacher_cfg.get('ENABLED', False)
        self.image_teacher_weight = teacher_cfg.get('LOSS_WEIGHT', 0.5)
        self.image_distill_weight = teacher_cfg.get('DISTILL_WEIGHT', 0.25)
        self.image_distill_obj_weight = teacher_cfg.get('OBJECT_WEIGHT', 2.0)
        self.image_teacher_channels = teacher_cfg.get('IMAGE_CHANNELS', 64)
        if self.use_image_teacher:
            self.image_encoder = LightImageEncoder(self.image_teacher_channels)
            self.image_teacher_interaction = DepthwiseInteraction(
                self.backbone_2d.num_bev_features + self.image_teacher_channels,
                self.backbone_2d.num_bev_features,
                num_layers=teacher_cfg.get('INTERACTION_LAYERS', 3)
            )
            self.image_teacher_head = DepthwiseObjectMapHead(
                self.backbone_2d.num_bev_features, out_channels=self.bev_object_map_classes
            )
            self.student_distill_adapter = DepthwiseInteraction(
                self.backbone_2d.num_bev_features,
                self.backbone_2d.num_bev_features,
                num_layers=1
            )

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            if cur_module is self.dense_head:
                batch_dict = self.apply_bev_object_fusion(batch_dict)
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def apply_bev_object_fusion(self, batch_dict):
        if not self.use_bev_object_map or 'spatial_features_2d' not in batch_dict:
            return batch_dict
        obj_logits = self.bev_object_head(batch_dict['spatial_features_2d'])
        batch_dict['bev_object_logits'] = obj_logits
        if self.use_bev_object_fusion:
            semantic_prob = obj_logits.sigmoid()
            interaction_input = torch.cat([batch_dict['spatial_features_2d'], semantic_prob], dim=1)
            interaction_feat = self.bev_object_interaction(interaction_input)
            batch_dict['spatial_features_2d'] = batch_dict['spatial_features_2d'] + (
                self.bev_object_fusion_scale * interaction_feat
            )
        return batch_dict

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size']
            # depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict
    
    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss = loss + loss_rpn
        
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(batch_dict, tb_dict)
            loss = loss + loss_backbone3d

        if self.model_cfg.get('POINT_HEAD', None) is not None:
            loss_point, tb_dict = self.point_head.get_loss()
            loss = loss + loss_point

        if self.use_bev_object_map and 'spatial_features_2d' in batch_dict:
            obj_logits = batch_dict.get('bev_object_logits', None)
            if obj_logits is None:
                obj_logits = self.bev_object_head(batch_dict['spatial_features_2d'])
            obj_target = self.build_bev_object_map(batch_dict, obj_logits.shape[-2], obj_logits.shape[-1], obj_logits.device)
            loss_obj = F.binary_cross_entropy_with_logits(obj_logits, obj_target)
            loss = loss + loss_obj * self.bev_object_map_weight
            tb_dict['loss_bev_object_map'] = loss_obj.item() * self.bev_object_map_weight

            if self.use_image_teacher and self.training and 'images' in batch_dict and 'trans_lidar_to_cam' in batch_dict:
                image_bev, image_mask = self.build_image_bev_teacher(batch_dict, obj_logits.shape[-2], obj_logits.shape[-1])
                teacher_input = torch.cat([batch_dict['spatial_features_2d'], image_bev], dim=1)
                teacher_feat = self.image_teacher_interaction(teacher_input)
                teacher_logits = self.image_teacher_head(teacher_feat)
                loss_teacher_obj = F.binary_cross_entropy_with_logits(teacher_logits, obj_target)

                student_feat = self.student_distill_adapter(batch_dict['spatial_features_2d'])
                student_norm = F.normalize(student_feat, dim=1)
                teacher_norm = F.normalize(teacher_feat.detach(), dim=1)
                obj_weight = 1.0 + self.image_distill_obj_weight * obj_target
                distill_weight = obj_weight * image_mask
                loss_distill = (
                    (student_norm - teacher_norm).abs() * distill_weight
                ).sum() / (distill_weight.sum() * student_norm.shape[1] + 1e-6)

                loss = loss + loss_teacher_obj * self.image_teacher_weight + loss_distill * self.image_distill_weight
                tb_dict['loss_image_teacher_obj'] = loss_teacher_obj.item() * self.image_teacher_weight
                tb_dict['loss_image_distill'] = loss_distill.item() * self.image_distill_weight

        return loss, tb_dict, disp_dict

    def build_image_bev_teacher(self, batch_dict, bev_h, bev_w):
        images = batch_dict['images']
        image_feat = self.image_encoder(images)
        batch_size, image_channels, feat_h, feat_w = image_feat.shape
        device = images.device
        image_bev = images.new_zeros((batch_size, image_channels, bev_h, bev_w))
        image_count = images.new_zeros((batch_size, 1, bev_h, bev_w))

        if 'fusion_voxel_coords' in batch_dict:
            voxel_coords = batch_dict['fusion_voxel_coords'].int()
        else:
            voxel_coords = batch_dict['x_indices'].int()
        voxel_centers = common_utils.get_voxel_centers(
            voxel_coords=voxel_coords[:, 1:4],
            downsample_times=1,
            voxel_size=self.dataset.voxel_size,
            point_cloud_range=self.dataset.point_cloud_range
        )

        ones = torch.ones((voxel_centers.shape[0], 1), device=device, dtype=voxel_centers.dtype)
        voxel_hom = torch.cat([voxel_centers, ones], dim=1)
        grid_size = torch.as_tensor(self.dataset.grid_size, device=device).float()
        stride_x = grid_size[0] / float(bev_w)
        stride_y = grid_size[1] / float(bev_h)
        bev_x = torch.clamp(torch.div(voxel_coords[:, 3].float(), stride_x, rounding_mode='floor').long(), 0, bev_w - 1)
        bev_y = torch.clamp(torch.div(voxel_coords[:, 2].float(), stride_y, rounding_mode='floor').long(), 0, bev_h - 1)

        padded_h, padded_w = images.shape[-2], images.shape[-1]
        image_shape = batch_dict.get('image_shape', None)
        for batch_idx in range(batch_size):
            mask = voxel_coords[:, 0] == batch_idx
            if mask.sum() == 0:
                continue
            pts = voxel_hom[mask]
            if 'lidar_aug_matrix' in batch_dict:
                aug_inv = torch.inverse(batch_dict['lidar_aug_matrix'][batch_idx])
                pts = (pts @ aug_inv.t())
            pts_cam = pts @ batch_dict['trans_lidar_to_cam'][batch_idx].t()
            pts_img_hom = pts_cam @ batch_dict['trans_cam_to_img'][batch_idx].t()
            depth = pts_img_hom[:, 2].clamp(min=1e-5)
            u = pts_img_hom[:, 0] / depth
            v = pts_img_hom[:, 1] / depth
            if image_shape is not None:
                cur_h = image_shape[batch_idx, 0].float()
                cur_w = image_shape[batch_idx, 1].float()
            else:
                cur_h = images.new_tensor(float(padded_h))
                cur_w = images.new_tensor(float(padded_w))
            valid = (depth > 0) & (u >= 0) & (u < cur_w) & (v >= 0) & (v < cur_h)
            if valid.sum() == 0:
                continue
            grid_u = u[valid] / max(float(padded_w - 1), 1.0) * 2.0 - 1.0
            grid_v = v[valid] / max(float(padded_h - 1), 1.0) * 2.0 - 1.0
            grid = torch.stack([grid_u, grid_v], dim=-1).view(1, -1, 1, 2)
            sampled = F.grid_sample(
                image_feat[batch_idx:batch_idx + 1], grid,
                mode='bilinear', padding_mode='zeros', align_corners=True
            ).squeeze(0).squeeze(-1).transpose(0, 1)
            flat_bev = bev_y[mask][valid] * bev_w + bev_x[mask][valid]
            image_bev_flat = image_bev[batch_idx].view(image_channels, -1)
            image_count_flat = image_count[batch_idx].view(1, -1)
            image_bev_flat.index_add_(1, flat_bev, sampled.transpose(0, 1))
            image_count_flat.index_add_(1, flat_bev, torch.ones((1, flat_bev.shape[0]), device=device, dtype=images.dtype))

        valid_mask = (image_count > 0).type_as(image_bev)
        image_bev = image_bev / image_count.clamp(min=1.0)
        return image_bev, valid_mask

    def build_bev_object_map(self, batch_dict, height, width, device):
        gt_boxes = batch_dict['gt_boxes']
        batch_size = gt_boxes.shape[0]
        target = gt_boxes.new_zeros((batch_size, self.bev_object_map_classes, height, width), device=device)
        point_cloud_range = torch.as_tensor(self.dataset.point_cloud_range, device=device).float()
        voxel_size = torch.as_tensor(self.dataset.voxel_size, device=device).float()
        grid_size = torch.as_tensor(self.dataset.grid_size, device=device).float()
        stride_x = grid_size[0] / float(width)
        stride_y = grid_size[1] / float(height)
        xs = torch.arange(width, device=device).float().view(1, width)
        ys = torch.arange(height, device=device).float().view(height, 1)

        for batch_idx in range(batch_size):
            cur_boxes = gt_boxes[batch_idx]
            valid = cur_boxes[:, 3] > 0
            cur_boxes = cur_boxes[valid]
            for box in cur_boxes:
                cls_idx = 0
                if box.shape[0] > 7:
                    cls_idx = int(box[-1].detach().clamp(min=1, max=self.bev_object_map_classes).item()) - 1
                cx = (box[0] - point_cloud_range[0]) / (voxel_size[0] * stride_x)
                cy = (box[1] - point_cloud_range[1]) / (voxel_size[1] * stride_y)
                if cx < 0 or cx >= width or cy < 0 or cy >= height:
                    continue
                sigma_x = torch.clamp(self.bev_gaussian_alpha * box[3] / (voxel_size[0] * stride_x), min=1.0)
                sigma_y = torch.clamp(self.bev_gaussian_alpha * box[4] / (voxel_size[1] * stride_y), min=1.0)
                gaussian = torch.exp(-0.5 * (((xs - cx) / sigma_x) ** 2 + ((ys - cy) / sigma_y) ** 2))
                target[batch_idx, cls_idx] = torch.maximum(target[batch_idx, cls_idx], gaussian)
        return target
