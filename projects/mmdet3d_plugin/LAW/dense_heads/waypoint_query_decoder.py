import torch, random
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from torch.nn import functional as F
import math

from mmdet3d.models import builder
from mmdet3d.models.builder import build_loss
from mmdet3d.core import AssignResult, PseudoSampler
from mmdet.core import build_bbox_coder, build_assigner, multi_apply, reduce_mean
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid

from projects.mmdet3d_plugin.LAW.utils import prj_pts_to_img

from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time, json

from projects.mmdet3d_plugin.LAW.dense_heads.utils import get_locations
from projects.mmdet3d_plugin.LAW.utils.visualization import prj_ego_traj_to_2d
# from thop import profile

@HEADS.register_module()
class WaypointHead(BaseModule):
    def __init__(self,
                num_proposals=6,
                #MHA
                hidden_channel=256,
                dim_feedforward=1024,
                num_heads=8,
                dropout=0.0,
                #pos embedding
                depth_step=0.8,
                depth_num=64,
                depth_start = 0,
                position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                stride=32,
                num_views=6,
                #others
                train_cfg=None,
                test_cfg=None,
                use_wm=True,
                num_spatial_token=36,
                num_tf_layers=2,
                num_traj_modal=1,
                **kwargs,
                ):
        """
        use to predict the waypoints
        """
        super().__init__(**kwargs)
        self.use_wm = use_wm

        # query feature
        self.num_views = num_views
        self.num_proposals = num_proposals
        self.view_query_feat = nn.Parameter(torch.randn(1, self.num_views, hidden_channel, self.num_proposals))
        self.waypoint_query_feat = nn.Parameter(torch.randn(1, self.num_proposals, hidden_channel))

        # spatial attn
        spatial_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        ) 
        
        self._spatial_decoder = nn.ModuleList( [
            nn.TransformerDecoder(spatial_decoder_layer, 1) 
            for _ in range(self.num_views)])

        # wp_attn
        wp_decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_channel,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        self.wp_attn = nn.TransformerDecoder(wp_decoder_layer, 1) # input: Bz, num_token, d_model

        # world model
        wm_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self._wm_decoder = nn.TransformerDecoder(wm_decoder_layer, num_tf_layers) 

        self.action_aware_encoder = nn.Sequential(
            nn.Linear(hidden_channel + 6*2, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
        )
        
        # loss
        self.loss_plan_reg = build_loss(dict(type='L1Loss', loss_weight=1.0))
        self.loss_plan_rec = nn.MSELoss()

        # head
        self.num_traj_modal = num_traj_modal
        self.waypoint_head = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, self.num_traj_modal* 2)
            )

        # position embedding
        ##img pos embed
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.depth_start = depth_start
        self.stride = stride

        self.position_encoder = nn.Sequential(
                nn.Linear(self.position_dim, hidden_channel*4),
                nn.ReLU(),
                nn.Linear(hidden_channel*4, hidden_channel),
            )

        self.pc_range = nn.Parameter(torch.tensor(point_cloud_range), requires_grad=False)
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        # LID depth
        index = torch.arange(start=0, end=self.depth_num, step=1).float()
        index_1 = index + 1
        bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
        coords_d = self.depth_start + bin_size * index * index_1
        self.coords_d = nn.Parameter(coords_d, requires_grad=False)


    def prepare_location(self, img_metas, img_feats):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = img_feats.shape[:2]
        x = img_feats.flatten(0, 1)
        location = get_locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location
    
    def img_position_embeding(self, img_feats, img_metas):
        """
        from streampetr
        """
        eps = 1e-5
        B, num_views, C, H, W = img_feats.shape
        assert num_views == self.num_views, 'num_views should be equal to self.num_views'
        BN = B * num_views
        num_sample_tokens = num_views * H * W
        LEN = num_sample_tokens
        img_pixel_locations = self.prepare_location(img_metas, img_feats)

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        img_pixel_locations[..., 0] = img_pixel_locations[..., 0] * pad_w
        img_pixel_locations[..., 1] = img_pixel_locations[..., 1] * pad_h

        # Depth
        D = self.coords_d.shape[0]
        pixel_centers = img_pixel_locations.detach().view(B, LEN, 1, 2).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([pixel_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        lidar2img = torch.from_numpy(np.stack(img_metas[0]['lidar2img'])).to(img_feats.device).float()
        lidar2img = lidar2img[:num_views]
        img2lidars = lidar2img.inverse()
        img2lidars = img2lidars.view(num_views, 1, 1, 4, 4).repeat(B, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3]) #normalize
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d) #(B, num_views*H*W, 3*64)
        coords_position_embeding = self.position_encoder(pos_embed)
        return coords_position_embeding
    
    def forward(self, img_feat, img_metas, ego_info=None, is_test=False):
        # init
        losses = {}
        Bz, num_views, num_channels, height, width = img_feat.shape
        init_view_query_feat = self.view_query_feat.clone().repeat(Bz, 1, 1, 1).permute(0, 1, 3, 2)
        init_waypoint_query_feat = self.waypoint_query_feat.clone().repeat(Bz, 1, 1)

        # img pos emb
        img_pos = self.img_position_embeding(img_feat, img_metas)
        img_pos = img_pos.reshape(Bz, num_views, height, width, num_channels)
        img_pos = img_pos.permute(0, 1, 4, 2, 3)
        img_feat_emb = img_feat + img_pos   

        # spatial view feat
        img_feat_emb = img_feat_emb.reshape(Bz, num_views, num_channels, height*width).permute(0, 1, 3, 2)
        spatial_view_feat = torch.zeros_like(init_view_query_feat)
        for i in range(self.num_views):
            spatial_view_feat[:, i] = self._spatial_decoder[i](init_view_query_feat[:, i], img_feat_emb[:, i])

        batch_size, num_view, num_tokens, num_channel = spatial_view_feat.shape
        spatial_view_feat = spatial_view_feat.reshape(batch_size, -1, num_channel)

        # predict wp
        updated_waypoint_query_feat = self.wp_attn(init_waypoint_query_feat, spatial_view_feat) #final_view_feat.shape torch.Size([1, 1440, 256])
        cur_waypoint = self.waypoint_head(updated_waypoint_query_feat)

        if self.num_traj_modal > 1:
            assert self.num_traj_modal == 3
            bz, traj_len, _ = cur_waypoint.shape
            cur_waypoint = cur_waypoint.reshape(bz, traj_len, self.num_traj_modal, 2)
            ego_cmd = img_metas[0]['ego_fut_cmd'].to(img_feat.device)[0, 0]
            cur_waypoint = cur_waypoint[: ,: ,ego_cmd == 1].squeeze(2)

        # world model prediction
        wm_next_latent = self.wm_prediction(spatial_view_feat, cur_waypoint)

        return cur_waypoint, spatial_view_feat, wm_next_latent
    
    def loss_reconstruction(self, 
            reconstructed_view_query_feat,
            observed_view_query_feat,
            ):
        loss_rec = self.loss_plan_rec(reconstructed_view_query_feat, observed_view_query_feat)
        return loss_rec
    
    def wm_prediction(self, view_query_feat, cur_waypoint):
        batch_size, num_tokens, num_channel = view_query_feat.shape
        cur_waypoint = cur_waypoint.reshape(batch_size, 1, -1).repeat(1, num_tokens, 1)
        cur_view_query_feat_with_ego = torch.cat([view_query_feat, cur_waypoint], dim=-1) 
        action_aware_latent = self.action_aware_encoder(cur_view_query_feat_with_ego)

        wm_next_latent = self._wm_decoder(action_aware_latent, action_aware_latent)
        return wm_next_latent
    
    def loss_3d(self, 
            preds_ego_future_traj,
            gt_ego_future_traj,
            gt_ego_future_traj_mask,
            ego_info=None,
            ):
        loss_waypoint = self.loss_plan_reg(preds_ego_future_traj, gt_ego_future_traj, gt_ego_future_traj_mask)
        return loss_waypoint
    

        
