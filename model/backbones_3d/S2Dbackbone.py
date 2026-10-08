from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.spconv_utils import replace_feature, spconv
from pcdet.utils.loss_utils import SigmoidFocalClassificationLoss
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from ...utils import common_utils, box_utils

from tools.svefusion_utils import SVSO, MambaBlock, Permute

from spconv.pytorch import functional as Fsp
from cumm.gemm.layout import to_stride
from typing import List
import numpy as np


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class S2DBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.input_channels = input_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        fusion_cfg = self.model_cfg.get('VOXEL_FUSION', {})
        self.voxel_fusion_type = fusion_cfg.get('TYPE', 'sparse_add')
        self.use_concat_mlp_fusion = self.voxel_fusion_type == 'concat_mlp'
        self.generated_confidence_min = fusion_cfg.get('GENERATED_CONFIDENCE_MIN', 0.05)
        if self.use_concat_mlp_fusion:
            fusion_hidden = fusion_cfg.get('HIDDEN_DIM', max(input_channels, 128))
            self.fusion_lidar_proj = nn.Linear(input_channels, input_channels, bias=False)
            self.fusion_radar_proj = nn.Linear(input_channels, input_channels, bias=False)
            self.fusion_generated_proj = nn.Linear(input_channels, input_channels, bias=False)
            self.fusion_lidar_norm = nn.LayerNorm(input_channels)
            self.fusion_radar_norm = nn.LayerNorm(input_channels)
            self.fusion_generated_norm = nn.LayerNorm(input_channels)
            self.fusion_lidar_embed = nn.Parameter(torch.zeros(1, input_channels))
            self.fusion_radar_embed = nn.Parameter(torch.zeros(1, input_channels))
            self.fusion_generated_embed = nn.Parameter(torch.zeros(1, input_channels))
            self.fusion_overlap_mlp = nn.Sequential(
                nn.Linear(input_channels * 2 + 2, fusion_hidden),
                nn.ReLU(),
                nn.Linear(fusion_hidden, input_channels)
            )
            nn.init.zeros_(self.fusion_overlap_mlp[-1].weight)
            nn.init.zeros_(self.fusion_overlap_mlp[-1].bias)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 64, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(64),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.conv2 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )
        self.conv3 = spconv.SparseSequential(
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        self.conv4 = spconv.SparseSequential(
            block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.svso = SVSO(features_in=256, down_mlp_channels=[256, 64, 1])
        self.pos_embed = nn.Sequential(
            nn.Linear(9, 128),
            Permute(0, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            Permute(0, 2, 1),
            nn.Linear(128, 256),
        )
        self.vmamba = nn.ModuleList([
            MambaBlock(
                d_model=256,
                ssm_cfg=None,
                norm_epsilon=0.00001,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                layer_idx=i,
                device='cuda',
                dtype=torch.float32)
            for i in range (self.model_cfg.NUM_MAMBA_LAYER)
        ])
        
        self.upconv1 = spconv.SparseSequential(
            block(256, 256, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv'),
        )
        self.upconv2 = spconv.SparseSequential(
            block(256, 128, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv'),
        )
        self.upconv3 = spconv.SparseSequential(
            block(128, 64, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv'),
        )
        
        self.seg_mlp64_1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.seg_mlp64_2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.seg_mlp128 = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.seg_mlp256_1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.seg_mlp256_2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.convupconv1 = spconv.SparseSequential(
            block(64, 64, kernel_size=(3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=(1, 0, 0), indice_key='spconv6', conv_type='spconv'),
        )
        self.convupconv2 = spconv.SparseSequential(
            block(64, 64, kernel_size=(3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=(1, 0, 0), indice_key='spconv7', conv_type='spconv'),
        )
        self.convupconv3 = spconv.SparseSequential(
            block(64, 64, kernel_size=(3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=(0, 0, 0), indice_key='spconv8', conv_type='spconv'),
        )
            
        last_pad = (0, 0, 0)
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        
        self.num_point_features = 64
        
        self.backbone_channels = {
            'x_conv1': 64,
            'x_conv2': 128,
            'x_conv3': 256,
            'x_conv4': 256
        }
        
        self.focal_loss = SigmoidFocalClassificationLoss()
        soft_cfg = self.model_cfg.get('SOFT_VOXEL_TARGET', {})
        self.use_soft_voxel_loss = self.model_cfg.get('USE_SOFT_VOXEL_LOSS', True)
        self.coverage_samples_per_axis = soft_cfg.get('SAMPLES_PER_AXIS', 3)
        self.coverage_low = soft_cfg.get('LOW', 0.4)
        self.coverage_high = soft_cfg.get('HIGH', 0.8)
        self.qfl_gamma = self.model_cfg.get('LOSS_CONFIG', {}).get('QFL_GAMMA', 2.0)
        self.dice_eps = 1e-6
        self.use_post_mamba_aux = self.model_cfg.get('USE_POST_MAMBA_AUX', True)
        if self.use_post_mamba_aux:
            self.post_mamba_score_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        sort_cfg = self.model_cfg.get('MAMBA_SORT', {})
        self.use_mamba_type_sort = sort_cfg.get('ENABLED', True)
        self.dual_conf_thresh = sort_cfg.get('DUAL_CONF_THRESH', 0.50)
        self.single_conf_thresh = sort_cfg.get('SINGLE_CONF_THRESH', 0.50)
        self.context_conf_margin = sort_cfg.get('CONTEXT_CONF_MARGIN', 0.10)
        
    def _indice_to_scalar(self, indices: torch.Tensor, shape: List[int]):
        assert indices.shape[1] == len(shape)
        stride = to_stride(np.array(shape, dtype=np.int64))
        scalar_inds = indices[:, -1].clone()
        for i in range(len(shape) - 1):
            scalar_inds += stride[i] * indices[:, i]
        return scalar_inds.contiguous()

    def _coord_hash(self, coords, batch_size):
        coords = coords.long()
        z_size, y_size, x_size = [int(v) for v in self.sparse_shape[:3]]
        batch_stride = z_size * y_size * x_size
        return coords[:, 0] * batch_stride + coords[:, 1] * (y_size * x_size) + coords[:, 2] * x_size + coords[:, 3]

    def _fuse_lidar_radar_features(self, batch_dict, batch_size):
        lidar_features = batch_dict['lidar_features']
        radar_features = batch_dict['radar_features']
        lidar_coords = batch_dict['lidar_voxel_coords'].int()
        radar_coords = batch_dict['radar_voxel_coords'].int()
        generated_features = batch_dict.get('generated_features', None)
        generated_coords = batch_dict.get('generated_voxel_coords', None)
        if generated_features is not None and generated_coords is not None:
            generated_coords = generated_coords.int()
            gen_conf = batch_dict.get('generated_confidence', None)
            if gen_conf is not None:
                keep_gen = gen_conf.view(-1) >= self.generated_confidence_min
                generated_features = generated_features[keep_gen]
                generated_coords = generated_coords[keep_gen]
                batch_dict['generated_features'] = generated_features
                batch_dict['generated_voxel_coords'] = generated_coords
                batch_dict['generated_confidence'] = gen_conf[keep_gen]

        if not self.use_concat_mlp_fusion:
            lidar_sp_tensor = spconv.SparseConvTensor(
                features=lidar_features,
                indices=lidar_coords,
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            radar_sp_tensor = spconv.SparseConvTensor(
                features=radar_features,
                indices=radar_coords,
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            if generated_features is not None and generated_features.shape[0] > 0:
                generated_sp_tensor = spconv.SparseConvTensor(
                    features=generated_features,
                    indices=generated_coords,
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                return Fsp.sparse_add(Fsp.sparse_add(lidar_sp_tensor, radar_sp_tensor), generated_sp_tensor)
            return Fsp.sparse_add(lidar_sp_tensor, radar_sp_tensor)

        coord_parts = [lidar_coords, radar_coords]
        if generated_features is not None and generated_features.shape[0] > 0:
            coord_parts.append(generated_coords)
        all_coords = torch.cat(coord_parts, dim=0)
        all_hash = self._coord_hash(all_coords, batch_size)
        sorted_hash, sorted_order = torch.sort(all_hash)
        keep = torch.ones_like(sorted_hash, dtype=torch.bool)
        keep[1:] = sorted_hash[1:] != sorted_hash[:-1]
        union_hash = sorted_hash[keep]
        union_coords = all_coords[sorted_order[keep]]

        lidar_hash = self._coord_hash(lidar_coords, batch_size)
        radar_hash = self._coord_hash(radar_coords, batch_size)
        lidar_pos = torch.searchsorted(union_hash, lidar_hash)
        radar_pos = torch.searchsorted(union_hash, radar_hash)
        if generated_features is not None and generated_features.shape[0] > 0:
            generated_hash = self._coord_hash(generated_coords, batch_size)
            generated_pos = torch.searchsorted(union_hash, generated_hash)
        else:
            generated_pos = None

        num_union = union_coords.shape[0]
        channels = lidar_features.shape[1]
        lidar_union = lidar_features.new_zeros((num_union, channels))
        radar_union = radar_features.new_zeros((num_union, channels))
        generated_union = lidar_features.new_zeros((num_union, channels))
        has_lidar = torch.zeros(num_union, dtype=torch.bool, device=lidar_features.device)
        has_radar = torch.zeros(num_union, dtype=torch.bool, device=radar_features.device)
        has_generated = torch.zeros(num_union, dtype=torch.bool, device=lidar_features.device)
        lidar_union[lidar_pos] = lidar_features
        radar_union[radar_pos] = radar_features
        has_lidar[lidar_pos] = True
        has_radar[radar_pos] = True
        if generated_pos is not None:
            generated_union[generated_pos] = generated_features
            has_generated[generated_pos] = True

        lidar_proj = self.fusion_lidar_norm(self.fusion_lidar_proj(lidar_union))
        radar_proj = self.fusion_radar_norm(self.fusion_radar_proj(radar_union))
        generated_proj = self.fusion_generated_norm(self.fusion_generated_proj(generated_union))

        if 'lidar_fg_probs' in batch_dict:
            lidar_prob_src = batch_dict['lidar_fg_probs'].type_as(lidar_features)
        else:
            lidar_prob_src = lidar_features.new_full((lidar_features.shape[0], 1), 0.5)
        if 'radar_fg_probs' in batch_dict:
            radar_prob_src = batch_dict['radar_fg_probs'].type_as(radar_features)
        else:
            radar_prob_src = radar_features.new_full((radar_features.shape[0], 1), 0.5)
        lidar_prob = lidar_features.new_zeros((num_union, 1))
        radar_prob = radar_features.new_zeros((num_union, 1))
        generated_conf = lidar_features.new_zeros((num_union, 1))
        lidar_prob[lidar_pos] = lidar_prob_src
        radar_prob[radar_pos] = radar_prob_src
        if generated_pos is not None:
            generated_conf[generated_pos] = batch_dict.get(
                'generated_confidence',
                lidar_features.new_zeros((generated_features.shape[0], 1))
            ).type_as(lidar_features)

        fused = lidar_features.new_zeros((num_union, channels))
        lidar_only = has_lidar & ~has_radar
        radar_only = has_radar & ~has_lidar
        overlap = has_lidar & has_radar
        generated_only = has_generated & ~has_lidar & ~has_radar
        fused[lidar_only] = lidar_proj[lidar_only] + self.fusion_lidar_embed
        fused[radar_only] = radar_proj[radar_only] + self.fusion_radar_embed
        if overlap.any():
            overlap_input = torch.cat([lidar_proj[overlap], radar_proj[overlap], lidar_prob[overlap], radar_prob[overlap]], dim=-1)
            # Start from projected summation and learn only a residual interaction.
            fused[overlap] = lidar_proj[overlap] + radar_proj[overlap] + self.fusion_overlap_mlp(overlap_input)
        if generated_only.any():
            fused[generated_only] = generated_proj[generated_only] + self.fusion_generated_embed
        generated_with_real = has_generated & (has_lidar | has_radar)
        if generated_with_real.any():
            fused[generated_with_real] = fused[generated_with_real] + generated_conf[generated_with_real] * generated_proj[generated_with_real]

        batch_dict['fusion_voxel_coords'] = union_coords
        batch_dict['fusion_has_lidar'] = has_lidar
        batch_dict['fusion_has_radar'] = has_radar
        batch_dict['fusion_has_generated'] = has_generated
        batch_dict['fusion_lidar_prob'] = lidar_prob
        batch_dict['fusion_radar_prob'] = radar_prob
        batch_dict['fusion_generated_confidence'] = generated_conf
        return spconv.SparseConvTensor(
            features=fused,
            indices=union_coords,
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

    def _morton_like_code(self, coords):
        coords = coords.long()
        return coords[:, 3] * 73856093 + coords[:, 2] * 19349663 + coords[:, 1] * 83492791

    def _lookup_mamba_sort_meta(self, target_coords, batch_dict, batch_size, stride=8):
        num_targets = target_coords.shape[0]
        groups = target_coords.new_full((num_targets,), 4, dtype=torch.long)
        confidence = target_coords.new_zeros((num_targets,), dtype=torch.float32)
        if 'fusion_voxel_coords' not in batch_dict:
            return groups, confidence

        fusion_coords = batch_dict['fusion_voxel_coords'].int().clone()
        if fusion_coords.shape[0] == 0:
            return groups, confidence
        fusion_down = fusion_coords.clone()
        fusion_down[:, 1:] = torch.div(fusion_down[:, 1:], stride, rounding_mode='floor')
        fusion_hash = self._coord_hash(fusion_down, batch_size)
        unique_hash, inverse = torch.unique(fusion_hash, sorted=True, return_inverse=True)
        num_unique = unique_hash.shape[0]
        device = target_coords.device

        def scatter_max(src, default=0.0):
            out = torch.full((num_unique, 1), default, device=device, dtype=torch.float32)
            out.scatter_reduce_(0, inverse.view(-1, 1), src.float(), reduce='amax', include_self=True)
            return out.view(-1)

        has_lidar = scatter_max(batch_dict.get('fusion_has_lidar').view(-1, 1).float()) > 0
        has_radar = scatter_max(batch_dict.get('fusion_has_radar').view(-1, 1).float()) > 0
        has_generated = scatter_max(batch_dict.get('fusion_has_generated').view(-1, 1).float()) > 0
        lidar_prob = scatter_max(batch_dict.get('fusion_lidar_prob', fusion_hash.new_zeros((fusion_hash.shape[0], 1))).float())
        radar_prob = scatter_max(batch_dict.get('fusion_radar_prob', fusion_hash.new_zeros((fusion_hash.shape[0], 1))).float())
        generated_conf = scatter_max(
            batch_dict.get('fusion_generated_confidence', fusion_hash.new_zeros((fusion_hash.shape[0], 1))).float()
        )

        target_hash = self._coord_hash(target_coords.int(), batch_size)
        pos = torch.searchsorted(unique_hash, target_hash)
        safe_pos = torch.clamp(pos, max=max(num_unique - 1, 0))
        found = (pos < num_unique) & (unique_hash[safe_pos] == target_hash)
        if not found.any():
            return groups, confidence

        idx = safe_pos[found]
        cur_has_lidar = has_lidar[idx]
        cur_has_radar = has_radar[idx]
        cur_has_generated = has_generated[idx]
        cur_lidar_prob = lidar_prob[idx]
        cur_radar_prob = radar_prob[idx]
        cur_generated_conf = generated_conf[idx]
        cur_conf = torch.maximum(torch.maximum(cur_lidar_prob, cur_radar_prob), cur_generated_conf)

        cur_groups = groups[found]
        context_thresh = max(float(self.single_conf_thresh) - float(self.context_conf_margin), 0.0)
        cur_groups[cur_conf >= context_thresh] = 3
        cur_groups[cur_has_generated] = 2
        single_reliable = (cur_has_lidar ^ cur_has_radar) & (cur_conf >= float(self.single_conf_thresh))
        cur_groups[single_reliable] = 1
        dual_reliable = cur_has_lidar & cur_has_radar & (torch.minimum(cur_lidar_prob, cur_radar_prob) >= float(self.dual_conf_thresh))
        cur_groups[dual_reliable] = 0

        groups[found] = cur_groups
        confidence[found] = cur_conf
        return groups, confidence

    def _build_mamba_sort_indices(self, coords, svso_scores, batch_dict, batch_size):
        if not self.use_mamba_type_sort or coords.shape[0] == 0:
            return svso_scores.argsort(descending=True)
        groups, confidence = self._lookup_mamba_sort_meta(coords, batch_dict, batch_size, stride=8)
        num_items = coords.shape[0]
        rank_base = num_items + 1
        score_order = svso_scores.float().argsort(descending=True)
        score_rank = torch.empty_like(score_order)
        score_rank[score_order] = torch.arange(num_items, device=coords.device, dtype=score_order.dtype)
        conf_order = confidence.float().argsort(descending=True)
        conf_rank = torch.empty_like(conf_order)
        conf_rank[conf_order] = torch.arange(num_items, device=coords.device, dtype=conf_order.dtype)
        morton_order = self._morton_like_code(coords).argsort(descending=False)
        morton_rank = torch.empty_like(morton_order)
        morton_rank[morton_order] = torch.arange(num_items, device=coords.device, dtype=morton_order.dtype)
        lex_key = (
            groups.long() * (rank_base ** 3) +
            score_rank.long() * (rank_base ** 2) +
            conf_rank.long() * rank_base +
            morton_rank.long()
        )
        return lex_key.argsort(descending=False)

    def _objectness_from_logits(self, logits):
        if logits.shape[-1] == 1:
            return logits.sigmoid()
        return logits.sigmoid().max(dim=-1, keepdim=True)[0]

    def forward(self, batch_dict):
        self.voxel_size = batch_dict['voxel_size']
        batch_size = batch_dict['batch_size']

        lidar_features = batch_dict['lidar_features']   # [M, 64]
        radar_features = batch_dict['radar_features']   # [N, 64]
        lidar_coords = batch_dict['lidar_voxel_coords'] # [M, 4(bzyx)]
        radar_coords = batch_dict['radar_voxel_coords'] # [N, 4(bzyx)]

        input_sp_tensor = self._fuse_lidar_radar_features(batch_dict, batch_size)

        # [41, 320, 320] * 64 channels
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x) # [41, 320, 320] * 64 channels
        x_conv2 = self.conv2(x_conv1)   # [21, 160, 160] * 128 channels
        x_conv3 = self.conv3(x_conv2)   # [11, 80, 80] * 256 channels
        x_conv4 = self.conv4(x_conv3)   # [5, 40, 40] * 256 channels
        
        x_conv4_mamba_recover = []
        for batch_idx in range(batch_size):
            batch_mask = x_conv4.indices[:, 0] == batch_idx
            x_conv4_batch_features = x_conv4.features[batch_mask]
            x_conv4_batch_coords = x_conv4.indices[batch_mask]
            _, _, svso_scores, mixed_embeddings = self.svso(x_conv4_batch_features.unsqueeze(0), return_aux=True)
            sorted_indices = self._build_mamba_sort_indices(
                x_conv4_batch_coords, svso_scores.squeeze(0), batch_dict, batch_size
            )
            sorted_embeddings = torch.gather(
                mixed_embeddings, 1,
                sorted_indices.view(1, -1, 1).expand(-1, -1, mixed_embeddings.shape[-1])
            )
            
            for mamba_block in self.vmamba:
                sorted_embeddings = mamba_block(sorted_embeddings, self.pos_embed)
            inv_indices = sorted_indices.argsort()
            x_conv4_mamba_recover.append(sorted_embeddings.squeeze(0).index_select(0, inv_indices))
            
        x_conv4_mamba_recover = torch.cat(x_conv4_mamba_recover, dim=0).index_select(0, x_conv4.indices[:, 0].argsort().argsort())
        x_conv4 = replace_feature(x_conv4, x_conv4.features * x_conv4_mamba_recover)    # [5, 40, 40] * 256 channels
        post_mamba_logits = self.post_mamba_score_head(x_conv4.features) if (self.training and self.use_post_mamba_aux) else None
        
        pred_logits_256_2 = self.seg_mlp256_2(x_conv4.features)
        x_conv4 = replace_feature(x_conv4, x_conv4.features * self._objectness_from_logits(pred_logits_256_2))

        x_upconv1 = self.upconv1(x_conv4)
        x_conv3 = replace_feature(x_conv3, x_conv3.features * x_upconv1.features)   # [11, 80, 80] * 256 channels
        pred_logits_256_1 = self.seg_mlp256_1(x_conv3.features)
        x_conv3 = replace_feature(x_conv3, x_conv3.features * self._objectness_from_logits(pred_logits_256_1))

        x_upconv2 = self.upconv2(x_conv3)
        x_conv2 = replace_feature(x_conv2, x_conv2.features * x_upconv2.features)   # [21, 160, 160] * 128 channels
        pred_logits_128 = self.seg_mlp128(x_conv2.features)
        x_conv2 = replace_feature(x_conv2, x_conv2.features * self._objectness_from_logits(pred_logits_128))
        
        x_upconv3 = self.upconv3(x_conv2)
        x_conv1 = replace_feature(x_conv1, x_conv1.features * x_upconv3.features)   # [41, 320, 320] * 64 channels
        pred_logits_64_2 = self.seg_mlp64_2(x_conv1.features)
        x_conv1 = replace_feature(x_conv1, x_conv1.features * self._objectness_from_logits(pred_logits_64_2))    
        
        x = replace_feature(x, x.features * x_conv1.features)   # [41, 320, 320] * 64 channels
        pred_logits_64_1 = self.seg_mlp64_1(x.features)
        x = replace_feature(x, x.features * self._objectness_from_logits(pred_logits_64_1))

        conv_x_conv1 = self.convupconv1(x)
        conv_x_conv2 = self.convupconv2(conv_x_conv1)
        conv_x_conv3 = self.convupconv3(conv_x_conv2)
        out = self.conv_out(conv_x_conv3)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 1,
            'x_indices': x.indices,
            'x_conv1_indices': x_conv1.indices,
            'x_conv2_indices': x_conv2.indices,
            'x_conv3_indices': x_conv3.indices,
            'x_conv4_indices': x_conv4.indices,
            'pred_logits_64_1': pred_logits_64_1,
            'pred_logits_64_2': pred_logits_64_2,
            'pred_logits_128': pred_logits_128,
            'pred_logits_256_1': pred_logits_256_1,
            'pred_logits_256_2': pred_logits_256_2,
        })
        if post_mamba_logits is not None:
            batch_dict['post_mamba_logits'] = post_mamba_logits
            batch_dict['post_mamba_indices'] = x_conv4.indices
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        return batch_dict

    def get_fg_labels(self, voxel_centers, voxel_coords, gt_boxes, voxel_size):
        bs_idx = voxel_coords[:, 0]
        batch_size = gt_boxes.shape[0]
        voxel_cls_labels = voxel_centers.new_zeros(voxel_centers.shape[0]).long()
        gt_extra_width = [voxel_size[i]/2 for i in range(3)]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=gt_extra_width
        ).view(batch_size, -1, gt_boxes.shape[-1])
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            voxels_single = voxel_centers[bs_mask]
            voxel_cls_labels_single = voxel_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = points_in_boxes_gpu(voxels_single.unsqueeze(0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            voxel_cls_labels_single[box_fg_flag] = 1
            voxel_cls_labels[bs_mask] = voxel_cls_labels_single
        return voxel_cls_labels

    def get_soft_fg_labels(self, voxel_centers, voxel_coords, gt_boxes, voxel_size):
        samples = int(self.coverage_samples_per_axis)
        if samples <= 1:
            return self.get_fg_labels(voxel_centers, voxel_coords, gt_boxes, voxel_size).float()

        device = voxel_centers.device
        voxel_size = torch.as_tensor(voxel_size, device=device, dtype=voxel_centers.dtype)
        lin = torch.linspace(-0.5, 0.5, samples + 2, device=device, dtype=voxel_centers.dtype)[1:-1]
        zz, yy, xx = torch.meshgrid(lin, lin, lin)
        offsets = torch.stack([xx, yy, zz], dim=-1).view(-1, 3) * voxel_size.view(1, 3)
        num_offsets = offsets.shape[0]

        bs_idx = voxel_coords[:, 0]
        labels = voxel_centers.new_zeros(voxel_centers.shape[0])
        for batch_idx in range(gt_boxes.shape[0]):
            mask = bs_idx == batch_idx
            if mask.sum() == 0:
                continue
            cur_centers = voxel_centers[mask]
            cur_points = (cur_centers[:, None, :] + offsets[None, :, :]).reshape(1, -1, 3)
            cur_gt = gt_boxes[batch_idx:batch_idx + 1, :, 0:7].contiguous()
            inside = points_in_boxes_gpu(cur_points, cur_gt).long().view(-1, num_offsets) >= 0
            coverage = inside.float().mean(dim=1)
            soft = ((coverage - self.coverage_low) / max(self.coverage_high - self.coverage_low, 1e-3)).clamp(0.0, 1.0)
            labels[mask] = soft
        return labels

    def quality_focal_loss(self, logits, target):
        target = target.type_as(logits)
        pred = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        return (target - pred).abs().pow(self.qfl_gamma) * bce

    def soft_dice_loss(self, logits, target):
        target = target.type_as(logits)
        pred = torch.sigmoid(logits)
        numerator = 2.0 * (pred * target).sum() + self.dice_eps
        denominator = pred.sum() + target.sum() + self.dice_eps
        return 1.0 - numerator / denominator

    def _voxel_seg_loss(self, logits, labels):
        if self.use_soft_voxel_loss:
            return self.quality_focal_loss(logits, labels).mean() + self.soft_dice_loss(logits, labels)
        return self.focal_loss(logits, labels.long(), torch.ones_like(labels.view(-1))).sum() / max(labels.shape[0], 1)

    def _downsample_coords(self, coords, stride):
        down_coords = coords.clone()
        down_coords[:, 1:] = torch.div(down_coords[:, 1:], stride, rounding_mode='floor')
        return down_coords

    def _lookup_post_probs(self, source_coords, post_coords, post_probs, stride):
        source_down = self._downsample_coords(source_coords.int(), stride)
        source_hash = self._coord_hash(source_down, int(source_coords[:, 0].max().item()) + 1)
        post_hash = self._coord_hash(post_coords.int(), int(post_coords[:, 0].max().item()) + 1)
        post_hash_sorted, order = torch.sort(post_hash)
        pos = torch.searchsorted(post_hash_sorted, source_hash)
        safe_pos = torch.clamp(pos, max=max(post_hash_sorted.shape[0] - 1, 0))
        found = (pos < post_hash_sorted.shape[0]) & (post_hash_sorted[safe_pos] == source_hash)
        teacher = post_probs.new_zeros((source_coords.shape[0], 1))
        if found.any():
            teacher[found] = post_probs[order[safe_pos[found]]].detach()
        return teacher, found

    def get_loss(self, batch_dict, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        voxel_centers_64_1 = common_utils.get_voxel_centers(
            voxel_coords=batch_dict['x_indices'][:, 1:4], 
            downsample_times=1, 
            voxel_size=self.voxel_size, 
            point_cloud_range=self.point_cloud_range
            )
        voxel_centers_64_2 = common_utils.get_voxel_centers(
            voxel_coords=batch_dict['x_conv1_indices'][:, 1:4], 
            downsample_times=1, 
            voxel_size=self.voxel_size, 
            point_cloud_range=self.point_cloud_range
            )
        voxel_centers_128 = common_utils.get_voxel_centers(
            voxel_coords=batch_dict['x_conv2_indices'][:, 1:4], 
            downsample_times=2, 
            voxel_size=self.voxel_size, 
            point_cloud_range=self.point_cloud_range
            )
        voxel_centers_256_1 = common_utils.get_voxel_centers(
            voxel_coords=batch_dict['x_conv3_indices'][:, 1:4], 
            downsample_times=4, 
            voxel_size=self.voxel_size, 
            point_cloud_range=self.point_cloud_range
            )
        voxel_centers_256_2 = common_utils.get_voxel_centers(
            voxel_coords=batch_dict['x_conv4_indices'][:, 1:4], 
            downsample_times=8, 
            voxel_size=self.voxel_size, 
            point_cloud_range=self.point_cloud_range
            )
        gt_boxes = batch_dict['gt_boxes']
        voxel_size_base = torch.as_tensor(self.voxel_size, device=voxel_centers_64_1.device).float()
        object_label_func = self.get_soft_fg_labels if self.use_soft_voxel_loss else self.get_fg_labels
        fg_labels_64_1 = object_label_func(
            voxel_centers_64_1, batch_dict['x_indices'], gt_boxes, voxel_size_base * 1
        ).view(-1, 1)
        fg_labels_64_2 = object_label_func(
            voxel_centers_64_2, batch_dict['x_conv1_indices'], gt_boxes, voxel_size_base * 1
        ).view(-1, 1)
        fg_labels_128 = object_label_func(
            voxel_centers_128, batch_dict['x_conv2_indices'], gt_boxes, voxel_size_base * 2
        ).view(-1, 1)
        fg_labels_256_1 = object_label_func(
            voxel_centers_256_1, batch_dict['x_conv3_indices'], gt_boxes, voxel_size_base * 4
        ).view(-1, 1)
        fg_labels_256_2 = object_label_func(
            voxel_centers_256_2, batch_dict['x_conv4_indices'], gt_boxes, voxel_size_base * 8
        ).view(-1, 1)
        post_object_labels_256_2 = object_label_func(
            voxel_centers_256_2, batch_dict['x_conv4_indices'], gt_boxes, voxel_size_base * 8
        ).view(-1, 1)
        
        loss_64_1_raw = self._voxel_seg_loss(batch_dict['pred_logits_64_1'], fg_labels_64_1)
        loss_64_2_raw = self._voxel_seg_loss(batch_dict['pred_logits_64_2'], fg_labels_64_2)
        loss_128_raw = self._voxel_seg_loss(batch_dict['pred_logits_128'], fg_labels_128)
        loss_256_1_raw = self._voxel_seg_loss(batch_dict['pred_logits_256_1'], fg_labels_256_1)
        loss_256_2_raw = self._voxel_seg_loss(batch_dict['pred_logits_256_2'], fg_labels_256_2)
        
        forward_ret_dict = {
            'loss_voxelseg_320v_1': loss_64_1_raw,
            'loss_voxelseg_320v_2': loss_64_2_raw,
            'loss_voxelseg_160v': loss_128_raw,
            'loss_voxelseg_80v': loss_256_1_raw,
            'loss_voxelseg_40v': loss_256_2_raw,
        }
        
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        
        loss_320p_1 = forward_ret_dict['loss_voxelseg_320v_1']
        loss_320p_2 = forward_ret_dict['loss_voxelseg_320v_2']
        loss_160p = forward_ret_dict['loss_voxelseg_160v']
        loss_80p = forward_ret_dict['loss_voxelseg_80v']
        loss_40p = forward_ret_dict['loss_voxelseg_40v']
        
        loss_320p_1 = loss_320p_1 * loss_weights_dict['voxelseg_layer_weight'][0]
        loss_320p_2 = loss_320p_2 * loss_weights_dict['voxelseg_layer_weight'][1]
        loss_160p = loss_160p * loss_weights_dict['voxelseg_layer_weight'][2]
        loss_80p = loss_80p * loss_weights_dict['voxelseg_layer_weight'][3]
        loss_40p = loss_40p * loss_weights_dict['voxelseg_layer_weight'][4]
        
        loss = loss_320p_1 + loss_320p_2 + loss_160p + loss_80p + loss_40p
        loss_post_fg = None
        loss_fg_cons = None

        if 'lidar_fg_logits' in batch_dict and 'radar_fg_logits' in batch_dict:
            lidar_centers = common_utils.get_voxel_centers(
                voxel_coords=batch_dict['lidar_voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            radar_centers = common_utils.get_voxel_centers(
                voxel_coords=batch_dict['radar_voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            lidar_fg_labels = object_label_func(
                lidar_centers, batch_dict['lidar_voxel_coords'], gt_boxes, voxel_size_base
            ).view(-1, 1)
            radar_fg_labels = object_label_func(
                radar_centers, batch_dict['radar_voxel_coords'], gt_boxes, voxel_size_base
            ).view(-1, 1)
            lidar_fg_loss = self._voxel_seg_loss(batch_dict['lidar_fg_logits'], lidar_fg_labels)
            radar_fg_loss = self._voxel_seg_loss(batch_dict['radar_fg_logits'], radar_fg_labels)
            loss_pre_fg = (lidar_fg_loss + radar_fg_loss) * loss_weights_dict.get('pre_fg_weight', 1.0)
            loss = loss + loss_pre_fg

            if 'post_mamba_logits' in batch_dict and 'post_mamba_indices' in batch_dict:
                post_fg_loss = self._voxel_seg_loss(batch_dict['post_mamba_logits'], post_object_labels_256_2)
                post_probs = torch.sigmoid(batch_dict['post_mamba_logits'])
                lidar_teacher, lidar_found = self._lookup_post_probs(
                    batch_dict['lidar_voxel_coords'], batch_dict['post_mamba_indices'], post_probs, stride=8
                )
                radar_teacher, radar_found = self._lookup_post_probs(
                    batch_dict['radar_voxel_coords'], batch_dict['post_mamba_indices'], post_probs, stride=8
                )
                loss_cons = batch_dict['post_mamba_logits'].new_tensor(0.0)
                cons_count = 0
                if lidar_found.any():
                    loss_cons = loss_cons + self.quality_focal_loss(
                        batch_dict['lidar_fg_logits'][lidar_found], lidar_teacher[lidar_found]
                    ).mean()
                    cons_count += 1
                if radar_found.any():
                    loss_cons = loss_cons + self.quality_focal_loss(
                        batch_dict['radar_fg_logits'][radar_found], radar_teacher[radar_found]
                    ).mean()
                    cons_count += 1
                if cons_count > 0:
                    loss_cons = loss_cons / cons_count
                loss_post_fg = post_fg_loss * loss_weights_dict.get('post_mamba_weight', 0.5)
                loss_fg_cons = loss_cons * loss_weights_dict.get('consistency_weight', 0.25)
                loss = loss + loss_post_fg + loss_fg_cons

        tb_dict['loss_voxelseg_320v_1'] = loss_320p_1.item()
        tb_dict['loss_voxelseg_320v_2'] = loss_320p_2.item()
        tb_dict['loss_voxelseg_160v'] = loss_160p.item()
        tb_dict['loss_voxelseg_80v'] = loss_80p.item()
        tb_dict['loss_voxelseg_40v'] = loss_40p.item()
        if 'lidar_fg_logits' in batch_dict and 'radar_fg_logits' in batch_dict:
            tb_dict['loss_pre_fg'] = loss_pre_fg.item()
        if loss_post_fg is not None and loss_fg_cons is not None:
            tb_dict['loss_post_fg'] = loss_post_fg.item()
            tb_dict['loss_fg_cons'] = loss_fg_cons.item()
        tb_dict['loss_voxelseg'] = loss.item()
        return loss, tb_dict
    
