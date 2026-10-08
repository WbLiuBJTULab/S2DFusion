import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from scipy.spatial import cKDTree

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate


class HeteroAlign(nn.Module):
    def __init__(self, lidar_dim, radar_dim, ada_dim=64, proj_dim=256):
        super().__init__()
        self.lidar_adapter  = nn.Sequential(
            nn.Linear(lidar_dim, ada_dim),
            nn.GELU()
        )
        self.radar_adapter  = nn.Linear(radar_dim, ada_dim)
        
        self.shared_proj  = nn.Linear(ada_dim, proj_dim)
        
    def forward(self, lidar, radar):
        lidar = self.lidar_adapter(lidar) 
        radar = self.radar_adapter(radar) 
        return self.shared_proj(lidar),  self.shared_proj(radar) 


class SNA(nn.Module):
    '''
    An attention mechanism that combines the information of two modalities 
    using neighborhood-based sparse attention.
    '''
    def __init__(self, uni_channels, channels):
        super(SNA, self).__init__()
        self.linear = nn.Linear(uni_channels, channels, bias=True)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, neighbors_idx):
        x = self.linear(x).permute(0, 2, 1)
        y = self.linear(y).permute(0, 2, 1)
        
        x_q = self.q_conv(x).permute(2, 0, 1)
        y_k = self.k_conv(y).permute(2, 1, 0)
        y_v = self.v_conv(y).permute(2, 0, 1)

        y_k_sparse = torch.index_select(y_k, 2, neighbors_idx.view(-1)).view(y_k.shape[0], y_k.shape[1], neighbors_idx.shape[0], neighbors_idx.shape[1]).permute(2, 0, 1, 3)
        y_v_sparse = torch.index_select(y_v, 1, neighbors_idx.view(-1)).view(y_v.shape[0], neighbors_idx.shape[0], neighbors_idx.shape[1], y_v.shape[2]).permute(1, 0, 2, 3)

        energy = torch.einsum('pmc,mpck->pmk', x_q, y_k_sparse)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        y_r = torch.einsum('pmk,mpkc->pmc', attention, y_v_sparse)
        y_r = y_r.permute(1, 2, 0)

        y_r = self.act(self.after_norm(self.trans_conv(x - y_r)))
        x = x + y_r
        x = torch.max(x, dim=2, keepdim=True)[0]
        return x
    

class Fusion(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features_l = num_point_features[0]
        num_point_features_r = num_point_features[1]
        self.use_decomposed_velocity = self.model_cfg.USE_DECOMPOSED_VELOCITY

        num_point_features_l += 6 if self.use_absolute_xyz else 3
        num_point_features_r += 5 if self.use_absolute_xyz else 2

        if self.with_distance:
            num_point_features_l += 1
            num_point_features_r += 1

        if self.use_decomposed_velocity:
            num_point_features_r += 8

        self.uniform_features = self.model_cfg.UNIFORM_FEATURES

        self.lr_align = HeteroAlign(num_point_features_l, num_point_features_r, self.uniform_features, self.uniform_features * 4)
        
        self.r2l_neighbor_num = self.model_cfg.R2L_NEIGHBOR_NUM
        self.l2r_neighbor_num = self.model_cfg.L2R_NEIGHBOR_NUM

        self.vfe_dim = self.model_cfg.VFE_DIM
        self.sna = SNA(self.uniform_features * 4, self.vfe_dim)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.num_point_features_r = num_point_features_r
        self.encode_transform = nn.Linear(num_point_features_r, num_point_features_r)

        self.use_fast_sparse = self.model_cfg.get('USE_FAST_SPARSE', False)
        if self.use_fast_sparse:
            fast_cfg = self.model_cfg.get('FAST_SPARSE', {})
            self.fast_channels = self.vfe_dim
            self.fast_dilations = fast_cfg.get('DILATIONS', [1, 3])
            self.fast_fg_ratio_min = fast_cfg.get('FOREGROUND_RATIO_MIN', 0.10)
            self.fast_fg_ratio_max = fast_cfg.get('FOREGROUND_RATIO_MAX', 0.35)
            self.fast_temperature = fast_cfg.get('FOREGROUND_TEMPERATURE', 0.10)
            self.use_radar_expansion = fast_cfg.get('USE_RADAR_EXPANSION', True)
            self.expansion_ratio = fast_cfg.get('EXPANSION_RATIO', 0.08)
            self.max_expansion_directions = fast_cfg.get('MAX_EXPANSION_DIRECTIONS', 2)
            self.max_expansion_radius = fast_cfg.get('MAX_EXPANSION_RADIUS', 3)

            self.lidar_voxel_proj = nn.Linear(self.uniform_features * 4, self.fast_channels)
            self.radar_voxel_proj = nn.Linear(self.uniform_features * 4, self.fast_channels)
            self.radar_to_lidar_proj = nn.Linear(self.fast_channels, self.fast_channels)
            self.lidar_to_radar_proj = nn.Linear(self.fast_channels, self.fast_channels)

            lr_attn_in = self.fast_channels * 2 + 3 + 2
            rl_attn_in = self.fast_channels * 2 + 3 + 2
            self.lr_attn = nn.Sequential(
                nn.Linear(lr_attn_in, self.fast_channels),
                nn.ReLU(),
                nn.Linear(self.fast_channels, 1)
            )
            self.rl_attn = nn.Sequential(
                nn.Linear(rl_attn_in, self.fast_channels),
                nn.ReLU(),
                nn.Linear(self.fast_channels, 1)
            )

            enhance_in = self.fast_channels * 3 + 2
            self.lidar_enhance = nn.Sequential(
                nn.Linear(enhance_in, self.fast_channels),
                nn.ReLU(),
                nn.Linear(self.fast_channels, self.fast_channels)
            )
            self.radar_enhance = nn.Sequential(
                nn.Linear(enhance_in, self.fast_channels),
                nn.ReLU(),
                nn.Linear(self.fast_channels, self.fast_channels)
            )
            nn.init.zeros_(self.lidar_enhance[-1].weight)
            nn.init.zeros_(self.lidar_enhance[-1].bias)
            nn.init.zeros_(self.radar_enhance[-1].weight)
            nn.init.zeros_(self.radar_enhance[-1].bias)

            self.lidar_fg_head = nn.Linear(self.fast_channels, 1)
            self.radar_fg_head = nn.Linear(self.fast_channels, 1)
            self.lidar_ratio_head = nn.Sequential(
                nn.Linear(self.fast_channels * 2, self.fast_channels),
                nn.ReLU(),
                nn.Linear(self.fast_channels, 1)
            )
            self.radar_ratio_head = nn.Sequential(
                nn.Linear(self.fast_channels * 2, self.fast_channels),
                nn.ReLU(),
                nn.Linear(self.fast_channels, 1)
            )
            if self.use_radar_expansion:
                self.radar_radius_head = nn.Linear(self.fast_channels + 2, 1)
                expansion_in = self.fast_channels + 1 + 2 + 1 + 2 + 1
                self.radar_expansion_head = nn.Sequential(
                    nn.Linear(expansion_in, self.fast_channels),
                    nn.ReLU(),
                    nn.Linear(self.fast_channels, 1)
                )
                gen_in = self.fast_channels + 2 + 1 + 1 + 1 + 1
                self.radar_generated_residual = nn.Sequential(
                    nn.Linear(gen_in, self.fast_channels),
                    nn.ReLU(),
                    nn.Linear(self.fast_channels, self.fast_channels * 2)
                )
                nn.init.zeros_(self.radar_generated_residual[-1].weight)
                nn.init.zeros_(self.radar_generated_residual[-1].bias)

            offsets = []
            for dilation in self.fast_dilations:
                cur_offsets = []
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            cur_offsets.append([dz * dilation, dy * dilation, dx * dilation])
                offsets.append(torch.tensor(cur_offsets, dtype=torch.long))
            self.register_buffer('fast_offsets', torch.stack(offsets, dim=0), persistent=False)
            bev_dirs = torch.tensor(
                [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]],
                dtype=torch.float32
            )
            self.register_buffer('bev_expansion_dirs', bev_dirs, persistent=False)
            self.register_buffer(
                'bev_expansion_radii',
                torch.arange(1, self.max_expansion_radius + 1, dtype=torch.float32),
                persistent=False
            )

    def get_output_feature_dim(self):
        return self.vfe_dim * len(self.r2l_neighbor_num)
    
    def time_embedding(self, t, embedding_dim):
        time_encoding = torch.zeros(t.shape[0], embedding_dim, device=t.device)

        for i, timestamp in enumerate(t):
            for j in range(0, embedding_dim, 2):
                time_encoding[i, j] = math.sin(timestamp * math.pow(10000, -j/embedding_dim))
                if j + 1 < embedding_dim:
                    time_encoding[i, j+1] = math.cos(timestamp * math.pow(10000, -(j+1)/embedding_dim))
        return time_encoding
    
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Args:
            actual_num: number of actual points per voxel
            max_num: the maximum number of voxel points
        Returns:
            paddings_indicator: Determine whether the data in the pillar is the real data or the filled value 0
        """

        # Extending a dimension
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def compute_voxel_neighbors(self, lidar_voxel_coords, radar_voxel_coords, k_neighbors=5):
        lidar_coords = lidar_voxel_coords[:, 1:]  # [M, 3]
        radar_coords = radar_voxel_coords[:, 1:]   # [N, 3]
        
        lidar_batch = lidar_voxel_coords[:, 0]     # [M]
        radar_batch = radar_voxel_coords[:, 0]     # [N]

        neighbors_idx = torch.full((len(lidar_coords), k_neighbors), -1, 
                            dtype=torch.long, device=lidar_voxel_coords.device)

        for batch_id in torch.unique(lidar_batch):
            # Mask for current batch
            lidar_mask = lidar_batch == batch_id
            radar_mask = radar_batch == batch_id
            
            radar_global_indices = torch.where(radar_mask)[0].cpu().numpy()
            
            batch_lidar = lidar_coords[lidar_mask].cpu().numpy()
            batch_radar = radar_coords[radar_mask].cpu().numpy()

            radar_tree = cKDTree(batch_radar)
            distances, local_indices = radar_tree.query(batch_lidar, k=k_neighbors)
            
            global_indices = np.full(local_indices.shape, -1)
            global_indices = radar_global_indices[local_indices]
            
            neighbors_idx[lidar_mask] = torch.tensor(global_indices, 
                                                device=lidar_voxel_coords.device)

        return neighbors_idx

    def _coord_hash(self, coords):
        coords = coords.long()
        x_size = int(self.grid_size[0].item())
        y_size = int(self.grid_size[1].item())
        z_size = int(self.grid_size[2].item())
        batch_stride = x_size * y_size * z_size
        return coords[:, 0] * batch_stride + coords[:, 1] * (y_size * x_size) + coords[:, 2] * x_size + coords[:, 3]

    def _query_fixed_neighbors(self, target_coords, source_coords, offsets):
        device = target_coords.device
        num_target = target_coords.shape[0]
        num_offsets = offsets.shape[0]
        if source_coords.shape[0] == 0:
            empty_idx = torch.zeros((num_target, num_offsets), dtype=torch.long, device=device)
            empty_mask = torch.zeros((num_target, num_offsets), dtype=torch.bool, device=device)
            return empty_idx, empty_mask

        source_hash = self._coord_hash(source_coords)
        source_hash_sorted, order = torch.sort(source_hash)

        query_coords = target_coords[:, None, :].long().repeat(1, num_offsets, 1)
        query_coords[:, :, 1:4] = query_coords[:, :, 1:4] + offsets[None, :, :]

        x_size = int(self.grid_size[0].item())
        y_size = int(self.grid_size[1].item())
        z_size = int(self.grid_size[2].item())
        in_range = (
            (query_coords[:, :, 1] >= 0) & (query_coords[:, :, 1] < z_size) &
            (query_coords[:, :, 2] >= 0) & (query_coords[:, :, 2] < y_size) &
            (query_coords[:, :, 3] >= 0) & (query_coords[:, :, 3] < x_size)
        )

        flat_query = query_coords.view(-1, 4)
        flat_hash = self._coord_hash(flat_query)
        pos = torch.searchsorted(source_hash_sorted, flat_hash)
        safe_pos = torch.clamp(pos, max=max(source_hash_sorted.shape[0] - 1, 0))
        found = (pos < source_hash_sorted.shape[0]) & (source_hash_sorted[safe_pos] == flat_hash) & in_range.view(-1)
        source_idx = order[safe_pos]
        source_idx = torch.where(found, source_idx, torch.zeros_like(source_idx))
        return source_idx.view(num_target, num_offsets), found.view(num_target, num_offsets)

    def _mean_valid_points(self, features, num_points):
        mask = self.get_paddings_indicator(num_points, features.shape[1], axis=0).unsqueeze(-1).type_as(features)
        summed = (features * mask).sum(dim=1)
        denom = num_points.clamp(min=1).type_as(features).view(-1, 1)
        return summed / denom

    def _aggregate_fixed(self, target_feats, target_coords, source_feats, source_coords, source_aux, offsets, attn_mlp, source_proj):
        neigh_idx, neigh_mask = self._query_fixed_neighbors(target_coords, source_coords, offsets)
        if source_feats.shape[0] == 0:
            zeros = target_feats.new_zeros(target_feats.shape)
            valid = target_feats.new_zeros((target_feats.shape[0], 1))
            return zeros, valid

        source_neigh = source_feats[neigh_idx]
        source_aux_neigh = source_aux[neigh_idx]
        target_neigh = target_feats[:, None, :].expand_as(source_neigh)
        delta = (source_coords[neigh_idx, 1:4].float() - target_coords[:, None, 1:4].float())
        delta = delta / max(float(torch.abs(offsets).max().item()), 1.0)
        attn_in = torch.cat([target_neigh, source_neigh, delta.type_as(target_neigh), source_aux_neigh], dim=-1)
        attn = torch.sigmoid(attn_mlp(attn_in)).squeeze(-1) * neigh_mask.type_as(target_feats)
        source_value = source_proj(source_neigh)
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        aggregated = (source_value * attn.unsqueeze(-1)).sum(dim=1) / denom
        valid = (neigh_mask.sum(dim=1, keepdim=True) > 0).type_as(target_feats)
        aggregated = aggregated * valid
        return aggregated, valid

    def _adaptive_suppress(self, feats, coords, logits, ratio_head):
        probs = torch.sigmoid(logits)
        weights = torch.zeros_like(probs)
        batch_ids = coords[:, 0]
        for batch_id in torch.unique(batch_ids):
            mask = batch_ids == batch_id
            cur_feats = feats[mask]
            cur_probs = probs[mask]
            if cur_probs.numel() == 0:
                continue
            pooled = torch.cat([cur_feats.mean(dim=0), cur_feats.max(dim=0)[0]], dim=0).unsqueeze(0)
            ratio = self.fast_fg_ratio_min + (self.fast_fg_ratio_max - self.fast_fg_ratio_min) * torch.sigmoid(ratio_head(pooled))
            kth = torch.clamp((cur_probs.shape[0] * (1.0 - ratio.detach())).long().view(-1), min=0, max=cur_probs.shape[0] - 1)
            threshold = torch.sort(cur_probs.detach().view(-1))[0][kth].view(1, 1)
            if self.training:
                gate = torch.sigmoid((cur_probs - threshold) / max(self.fast_temperature, 1e-3))
                cur_weights = gate + (1.0 - gate) * cur_probs
            else:
                hard = (cur_probs >= threshold).type_as(cur_probs)
                cur_weights = hard + (1.0 - hard) * cur_probs
            weights[mask] = cur_weights
        return probs, weights

    def _build_radar_expansion(self, radar_feats, radar_coords, radar_probs, radar_weights, radar_aux):
        if (not self.use_radar_expansion) or radar_feats.shape[0] == 0:
            return None

        device = radar_feats.device
        dirs = self.bev_expansion_dirs.to(device=device, dtype=radar_feats.dtype)
        radii = self.bev_expansion_radii.to(device=device, dtype=radar_feats.dtype)
        num_dirs = dirs.shape[0]
        num_radii = radii.shape[0]
        if num_dirs == 0 or num_radii == 0:
            return None

        rcs = radar_aux[:, 0:1]
        vel = radar_aux[:, 1:2]
        abs_vel = vel.abs()
        radar_fg = (radar_weights >= 0.5).view(-1)

        center_x = radar_coords[:, 3].type_as(radar_feats) * self.voxel_x + self.x_offset
        center_y = radar_coords[:, 2].type_as(radar_feats) * self.voxel_y + self.y_offset
        radial = torch.stack([center_x, center_y], dim=-1)
        radial = radial / radial.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        radial = radial * vel.sign().clamp(min=-1.0, max=1.0)
        dir_norm = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        align = torch.matmul(radial, dir_norm.t()).view(-1, num_dirs, 1)
        dir_keep = torch.ones((radar_feats.shape[0], num_dirs, 1), device=device, dtype=torch.bool)
        max_dirs = min(max(int(self.max_expansion_directions), 1), num_dirs)
        if max_dirs < num_dirs:
            dir_score = align.squeeze(-1)
            _, keep_dir_idx = torch.topk(dir_score, k=max_dirs, dim=1, largest=True, sorted=False)
            dir_keep = torch.zeros((radar_feats.shape[0], num_dirs), device=device, dtype=torch.bool)
            dir_keep.scatter_(1, keep_dir_idx, True)
            dir_keep = dir_keep.unsqueeze(-1)

        radius_score = torch.sigmoid(self.radar_radius_head(torch.cat([radar_feats, radar_probs, abs_vel], dim=-1)))
        radius_limit = 1 + torch.round((num_radii - 1) * radius_score).long().clamp(min=0, max=max(num_radii - 1, 0))
        radius_valid = radii.view(1, 1, num_radii) <= radius_limit.view(-1, 1, 1).type_as(radii)
        radius_valid = radius_valid & dir_keep

        feat_expand = radar_feats[:, None, None, :].expand(-1, num_dirs, num_radii, -1)
        prob_expand = radar_probs[:, None, None, :].expand(-1, num_dirs, num_radii, -1)
        aux_expand = torch.cat([rcs, abs_vel], dim=-1)[:, None, None, :].expand(-1, num_dirs, num_radii, -1)
        align_expand = align[:, :, None, :].expand(-1, -1, num_radii, -1)
        dir_expand = dirs.view(1, num_dirs, 1, 2).expand(radar_feats.shape[0], -1, num_radii, -1)
        radius_expand = (radii / max(float(self.max_expansion_radius), 1.0)).view(1, 1, num_radii, 1).expand(
            radar_feats.shape[0], num_dirs, -1, -1
        )
        expansion_input = torch.cat(
            [feat_expand, prob_expand, aux_expand, align_expand, dir_expand, radius_expand], dim=-1
        )
        expansion_prob = torch.sigmoid(self.radar_expansion_head(expansion_input)).squeeze(-1)
        candidate_scores = expansion_prob * radar_probs.view(-1, 1, 1)
        candidate_scores = candidate_scores.masked_fill(~radius_valid, -1.0)
        candidate_scores = candidate_scores.masked_fill(~radar_fg.view(-1, 1, 1), -1.0)

        coord_base = radar_coords[:, None, None, :].long().expand(-1, num_dirs, num_radii, -1).clone()
        steps = dirs.view(1, num_dirs, 1, 2) * radii.view(1, 1, num_radii, 1)
        coord_base[:, :, :, 2] = coord_base[:, :, :, 2] + steps[:, :, :, 1].long()
        coord_base[:, :, :, 3] = coord_base[:, :, :, 3] + steps[:, :, :, 0].long()
        x_size = int(self.grid_size[0].item())
        y_size = int(self.grid_size[1].item())
        in_range = (
            (coord_base[:, :, :, 2] >= 0) & (coord_base[:, :, :, 2] < y_size) &
            (coord_base[:, :, :, 3] >= 0) & (coord_base[:, :, :, 3] < x_size)
        )
        candidate_scores = candidate_scores.masked_fill(~in_range, -1.0)

        selected_coords, selected_src, selected_dirs, selected_radii, selected_conf = [], [], [], [], []
        flat_scores = candidate_scores.view(radar_feats.shape[0], -1)
        batch_ids = radar_coords[:, 0]
        for batch_id in torch.unique(batch_ids):
            radar_mask = batch_ids == batch_id
            num_radar = int(radar_mask.sum().item())
            budget = min(max(int(math.ceil(num_radar * float(self.expansion_ratio))), 0), flat_scores[radar_mask].numel())
            if budget <= 0:
                continue
            cur_scores = flat_scores[radar_mask].reshape(-1)
            valid = cur_scores > 0
            if not valid.any():
                continue
            budget = min(budget, int(valid.sum().item()))
            top_scores, top_idx = torch.topk(cur_scores, k=budget, largest=True, sorted=False)
            local_radar_idx = top_idx // (num_dirs * num_radii)
            rem = top_idx % (num_dirs * num_radii)
            dir_idx = rem // num_radii
            radius_idx = rem % num_radii
            global_radar_idx = torch.where(radar_mask)[0][local_radar_idx]
            selected_coords.append(coord_base[global_radar_idx, dir_idx, radius_idx])
            selected_src.append(radar_coords[global_radar_idx])
            selected_dirs.append(dirs[dir_idx])
            selected_radii.append(radii[radius_idx].view(-1, 1))
            selected_conf.append(top_scores.view(-1, 1))

        if len(selected_coords) == 0:
            return None

        gen_coords = torch.cat(selected_coords, dim=0).int()
        gen_src_coords = torch.cat(selected_src, dim=0).int()
        gen_dirs = torch.cat(selected_dirs, dim=0).type_as(radar_feats)
        gen_radii = torch.cat(selected_radii, dim=0).type_as(radar_feats)
        gen_conf = torch.cat(selected_conf, dim=0).type_as(radar_feats)

        src_hash = self._coord_hash(gen_src_coords)
        radar_hash = self._coord_hash(radar_coords)
        radar_hash_sorted, radar_order = torch.sort(radar_hash)
        src_pos = torch.searchsorted(radar_hash_sorted, src_hash)
        src_idx = radar_order[src_pos.clamp(max=radar_hash_sorted.shape[0] - 1)]
        src_feat = radar_feats[src_idx]
        src_aux = radar_aux[src_idx]

        gen_input = torch.cat([src_feat, gen_dirs, gen_radii / max(float(self.max_expansion_radius), 1.0),
                               src_aux[:, 1:2], src_aux[:, 0:1], gen_conf], dim=-1)
        gen_features = torch.cat([src_feat, src_feat * gen_conf], dim=-1) + self.radar_generated_residual(gen_input)

        gen_hash = self._coord_hash(gen_coords)
        unique_hash, inverse = torch.unique(gen_hash, sorted=True, return_inverse=True)
        order = torch.argsort(gen_hash)
        keep = torch.ones_like(order, dtype=torch.bool)
        keep[1:] = gen_hash[order][1:] != gen_hash[order][:-1]
        unique_keep_order = order[keep]
        unique_coords = gen_coords[unique_keep_order]

        conf_sum = gen_conf.new_zeros((unique_hash.shape[0], 1))
        feat_sum = gen_features.new_zeros((unique_hash.shape[0], gen_features.shape[1]))
        conf_sum.index_add_(0, inverse, gen_conf)
        feat_sum.index_add_(0, inverse, gen_features * gen_conf)
        unique_features = feat_sum / conf_sum.clamp(min=1e-6)
        max_conf = gen_conf.new_full((unique_hash.shape[0], 1), -1.0)
        max_conf.scatter_reduce_(0, inverse.view(-1, 1), gen_conf, reduce='amax', include_self=True)

        return {
            'coords': unique_coords,
            'features': unique_features,
            'confidence': max_conf.clamp(min=0.0),
            'source_coords': gen_src_coords[unique_keep_order],
            'directions': gen_dirs[unique_keep_order],
            'radii': gen_radii[unique_keep_order],
            'voxel_type': gen_coords.new_full((unique_coords.shape[0], 1), 2)
        }

    def _forward_fast_sparse(self, batch_dict, lidar_features, radar_features, lidar_coords, radar_coords, radar_voxel_features, radar_voxel_num_points):
        lidar_voxel_features = self._mean_valid_points(lidar_features, batch_dict['lidar_voxel_num_points'])
        radar_voxel_features = self._mean_valid_points(radar_features, radar_voxel_num_points)

        lidar_voxel_features = self.lidar_voxel_proj(lidar_voxel_features)
        radar_voxel_features = self.radar_voxel_proj(radar_voxel_features)

        if batch_dict['radar_voxels'].shape[-1] >= 5:
            raw_radar_aux = self._mean_valid_points(batch_dict['radar_voxels'][:, :, 3:5].type_as(radar_voxel_features), radar_voxel_num_points)
        else:
            raw_radar_aux = radar_voxel_features.new_zeros((radar_voxel_features.shape[0], 2))
        lidar_aux = lidar_voxel_features.new_zeros((lidar_voxel_features.shape[0], 2))

        lidar_aggs, lidar_valids = [], []
        radar_aggs, radar_valids = [], []
        for scale_idx in range(len(self.fast_dilations)):
            offsets = self.fast_offsets[scale_idx].to(lidar_coords.device)
            agg_l, valid_l = self._aggregate_fixed(
                lidar_voxel_features, lidar_coords, radar_voxel_features, radar_coords, raw_radar_aux,
                offsets, self.lr_attn, self.radar_to_lidar_proj
            )
            agg_r, valid_r = self._aggregate_fixed(
                radar_voxel_features, radar_coords, lidar_voxel_features, lidar_coords, lidar_aux,
                offsets, self.rl_attn, self.lidar_to_radar_proj
            )
            lidar_aggs.append(agg_l)
            lidar_valids.append(valid_l)
            radar_aggs.append(agg_r)
            radar_valids.append(valid_r)

        lidar_enhance_in = torch.cat([lidar_voxel_features, lidar_aggs[0], lidar_aggs[-1], lidar_valids[0], lidar_valids[-1]], dim=-1)
        radar_enhance_in = torch.cat([radar_voxel_features, radar_aggs[0], radar_aggs[-1], radar_valids[0], radar_valids[-1]], dim=-1)
        lidar_enhanced = lidar_voxel_features + self.lidar_enhance(lidar_enhance_in)
        radar_enhanced = radar_voxel_features + self.radar_enhance(radar_enhance_in)

        lidar_logits = self.lidar_fg_head(lidar_enhanced)
        radar_logits = self.radar_fg_head(radar_enhanced)
        lidar_probs, lidar_weights = self._adaptive_suppress(lidar_enhanced, lidar_coords, lidar_logits, self.lidar_ratio_head)
        radar_probs, radar_weights = self._adaptive_suppress(radar_enhanced, radar_coords, radar_logits, self.radar_ratio_head)

        lidar_suppressed = lidar_enhanced * lidar_weights
        radar_suppressed = radar_enhanced * radar_weights
        radar_generated = self._build_radar_expansion(
            radar_enhanced, radar_coords, radar_probs, radar_weights, raw_radar_aux
        )

        batch_dict['lidar_features'] = torch.cat([lidar_enhanced, lidar_suppressed], dim=-1)
        batch_dict['radar_features'] = torch.cat([radar_enhanced, radar_suppressed], dim=-1)
        batch_dict['lidar_fg_logits'] = lidar_logits
        batch_dict['radar_fg_logits'] = radar_logits
        batch_dict['lidar_fg_probs'] = lidar_probs
        batch_dict['radar_fg_probs'] = radar_probs
        batch_dict['lidar_voxel_coords'] = lidar_coords
        batch_dict['radar_voxel_coords'] = radar_coords
        if radar_generated is not None:
            batch_dict['generated_features'] = radar_generated['features']
            batch_dict['generated_voxel_coords'] = radar_generated['coords']
            batch_dict['generated_confidence'] = radar_generated['confidence']
            batch_dict['generated_source_coords'] = radar_generated['source_coords']
            batch_dict['generated_directions'] = radar_generated['directions']
            batch_dict['generated_radii'] = radar_generated['radii']
            batch_dict['generated_voxel_type'] = radar_generated['voxel_type']
        batch_dict['voxel_size'] = self.voxel_size
        return batch_dict

    def forward(self, batch_dict, **kwargs):
        lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
        radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']

        orig_xyz_l = lidar_voxel_features[:, :, :3]  # selecting x y z
        orig_xyz_r = radar_voxel_features[:, :, :3]  # selecting x y z

        lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
        radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
        lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
        radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

        lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
        radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
        lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.with_distance:
            points_dist_l = torch.norm(orig_xyz_l, 2, 2, keepdim=True)
            points_dist_r = torch.norm(orig_xyz_r, 2, 2, keepdim=True)

        if self.use_decomposed_velocity:
            v_r = radar_voxel_features[:, :, 4]
            v_r_compensated = radar_voxel_features[:, :, 5]

            beta = torch.atan2(radar_voxel_features[:, :, 1], radar_voxel_features[:, :, 0])  # β = arctan(y / x)
            v_x = torch.cos(beta) * v_r
            v_y = torch.sin(beta) * v_r
            v_x_compensated = torch.cos(beta) * v_r_compensated
            v_y_compensated = torch.sin(beta) * v_r_compensated

            v_features = torch.stack([v_x, v_y, v_x_compensated, v_y_compensated], dim=-1)
            v_features_mean = v_features.sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            v_features_diff = v_features - v_features_mean

        if self.use_absolute_xyz:
            lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            radar_features = [radar_voxel_features[:, :, :-1], radar_f_cluster, radar_f_center]
        else:
            lidar_features = [lidar_voxel_features[:, :, 3:], lidar_f_cluster, lidar_f_center]
            radar_features = [radar_voxel_features[:, :, 3:-1], radar_f_cluster, radar_f_center]

        if self.with_distance:
            lidar_features.append(points_dist_l)
            radar_features.append(points_dist_r)
        
        if self.use_decomposed_velocity:
            radar_features.append(v_features)
            radar_features.append(v_features_diff)

        lidar_features = torch.cat(lidar_features, dim=-1)
        radar_features = torch.cat(radar_features, dim=-1)

        # cosine embedding for time
        t = [-4.0, -3.0, -2.0, -1.0, 0.0]
        t = torch.tensor(t, dtype=torch.float32).cuda()
        time_encoding = self.time_embedding(t, self.num_point_features_r)
        time_encoding = self.encode_transform(time_encoding)

        time = radar_voxel_features[:, :, -1].squeeze(-1).view(-1)
        time_encoding = time_encoding[time.int() + 4].view(radar_features.shape[0], radar_features.shape[1], -1)
        radar_features = radar_features + time_encoding

        lidar_features, radar_features = self.lr_align(lidar_features, radar_features)

        lidar_voxel_count = lidar_features.shape[1]
        radar_voxel_count = radar_features.shape[1]
        lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
        radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
        lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
        radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
        lidar_features *= lidar_mask
        radar_features *= radar_mask

        if self.use_fast_sparse:
            return self._forward_fast_sparse(
                batch_dict, lidar_features, radar_features, lidar_coords, radar_coords,
                radar_voxel_features, radar_voxel_num_points
            )

        assert len(self.r2l_neighbor_num) == len(self.l2r_neighbor_num)
        lidar_features_output_list = []
        radar_features_output_list = []
        for i in range(len(self.r2l_neighbor_num)):
            neighbors_idx = self.compute_voxel_neighbors(lidar_coords, radar_coords, k_neighbors=self.r2l_neighbor_num[i])
            lidar_features_output = self.sna(lidar_features, radar_features, neighbors_idx)
            lidar_features_output = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            lidar_features_output_list.append(lidar_features_output)

            neighbors_idx = self.compute_voxel_neighbors(radar_coords, lidar_coords, k_neighbors=self.l2r_neighbor_num[i])
            radar_features_output = self.sna(radar_features, lidar_features, neighbors_idx)
            radar_features_output = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])
            radar_features_output_list.append(radar_features_output)

        lidar_features_output = torch.cat(lidar_features_output_list, dim=-1)
        radar_features_output = torch.cat(radar_features_output_list, dim=-1)

        batch_dict['voxel_size'] = self.voxel_size

        batch_dict['lidar_features'] = lidar_features_output
        batch_dict['radar_features'] = radar_features_output

        return batch_dict
