import time
import copy
import numpy as np
import torch, cv2, os, random
import torch.nn as nn
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin import VAD
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric
from mmdet3d.models import builder

from projects.mmdet3d_plugin.LAW.utils import prj_pts_to_img, draw_lidar_pts, denormalize_img
import matplotlib.pyplot as plt
from ipdb import set_trace

@DETECTORS.register_module()
class LAW(VAD):
    def __init__(self,
                use_video=False,
                use_swin=False,
                only_front_view=False,
                use_multi_view=True,
                swin_input_channel=768,
                hidden_channel=256,
                use_semantic=False,
                semantic_img_backbone=None,
                flow_only=False,
                all_zero=False,
                semantic_only=False,
                use_2d_waypoint=False,
                wm_loss_weight=0.2,
                **kwargs,
                 ):
        super().__init__( **kwargs)
        self.depth_eps = 2.75
        self.ref_pts_cam_list = []
        self.use_video = use_video
        self.use_swin = use_swin
        self.use_semantic = use_semantic

        self.flow_only = flow_only
        self.semantic_only = semantic_only
        self.all_zero = all_zero

        self.only_front_view = only_front_view
        self.use_2d_waypoint = use_2d_waypoint

        if semantic_img_backbone is not None:
            self.semantic_img_backbone = builder.build_backbone(semantic_img_backbone)

        if (not self.with_img_neck) and self.use_swin:
            self.swin_img_mlp = nn.Linear(swin_input_channel, hidden_channel)
        
        self.metrics_history = []
        self.call_count = 0
        self.wm_loss_weight = wm_loss_weight
        

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            # My modification
            if self.use_video:
                img = img.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
                img_feats = self.img_backbone(img)
                img_feats = img_feats.mean(dim=2, keepdim=True)
            else:
                if self.use_swin:
                    img = img.unsqueeze(2)
                img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        if not isinstance(img_feats, tuple):
            img_feats = [img_feats.squeeze(2)]

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        elif self.use_swin: #swin without fpn
            img_feats = [self.swin_img_mlp(img_feats[0].permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()]

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    
    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def obtain_history_feat(self, imgs_queue, img_metas_list, is_test=False):
        """Obtain history BEV features iteratively.
        """
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape
        imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
        img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
        losses = {}
        for i in range(len_queue):
            img_metas = [each[i] for each in img_metas_list]
            img_feats = [each_scale[:, i] for each_scale in img_feats_list][0]
            pred_ego_fut_trajs, cur_img_feat, pred_img_feat = self.pts_bbox_head(img_feats, img_metas)
            
            # compute loss
            if not is_test:
                # loss waypoint
                gt_ego_fut_trajs = img_metas[0]['ego_fut_trajs'].to(img_feats.device)
                gt_ego_fut_masks = img_metas[0]['ego_fut_masks'].squeeze(0).unsqueeze(-1).to(img_feats.device)
                loss_waypoint = self.pts_bbox_head.loss_3d(pred_ego_fut_trajs,
                                                            gt_ego_fut_trajs,
                                                            gt_ego_fut_masks,
                                                            )
                losses.update({
                    f'prev_frame_loss_waypoint_{i}': loss_waypoint
                })

        return losses, pred_img_feat

    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      semantic_img=None,
                      flow_img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None
                      ):
        """
        agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
        """
        if self.only_front_view:
            img = img[:, :, 0:1, ...]

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)

        self.pts_bbox_head.prev_view_feat = None
        if len_queue > 1:
            prev_frame_losses, pred_img_feat = self.obtain_history_feat(prev_img, prev_img_metas)  
        else:
            prev_frame_losses = {}

        cur_img = img[:, -1, ...]
        cur_img_metas = [each[len_queue-1] for each in img_metas]

        cur_img_feats = self.extract_feat(img=cur_img, img_metas=cur_img_metas)[0]            
        losses = self.forward_pts_train(cur_img_feats, 
                                        cur_img_metas,
                                        pred_img_feat=pred_img_feat,
                                        ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                        ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                        ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels,
                                    )
        losses.update(prev_frame_losses)
        return losses

    def forward_pts_train(self,
                          img_feats,
                          img_metas,
                          pred_img_feat=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None,
                        ):
        """Forward function
        Args:
            ego_fut_cmd: [turn_left, turn_right, go_straight]
            ego_lcf_feat: (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度

        """
        #get the ego info   
        losses = {}
        B = ego_his_trajs.size(0)
        ego_his_trajs = ego_his_trajs.reshape(B, -1)
        ego_lcf_feat = ego_lcf_feat.reshape(B, -1)
        ego_fut_cmd = ego_fut_cmd.reshape(B, -1)
        ego_info = torch.cat([ego_his_trajs, ego_lcf_feat, ego_fut_cmd], dim=1)
        
        prev_pred_img_feat = pred_img_feat
        preds_ego_future_traj, cur_img_feat, pred_img_feat = self.pts_bbox_head(img_feats, img_metas, ego_info)
        
        # world model loss
        loss_rec = self.pts_bbox_head.loss_reconstruction(
                                prev_pred_img_feat, 
                                cur_img_feat.detach(),
                                )
        losses['loss_rec'] = loss_rec * self.wm_loss_weight

        # waypoint loss
        loss_waypoint = self.pts_bbox_head.loss_3d(preds_ego_future_traj,
                                            ego_fut_trajs.squeeze(1),
                                            ego_fut_masks.squeeze(0).squeeze(0).unsqueeze(-1),
                                            )
        losses.update({
            'loss_waypoint': loss_waypoint
        })
        return losses

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        bbox_results = self.simple_test(
            img_metas=img_metas,
            img=img,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            **kwargs
        )

        return bbox_results
    
    def simple_test(
        self,
        img_metas,
        img,
        gt_bboxes_3d,
        gt_labels_3d,
        fut_valid_flag=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs,
    ):
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        self.pts_bbox_head.prev_view_feat = None
        if len_queue > 1:
            _ = self.obtain_history_feat(prev_img, prev_img_metas, is_test=True)  

        cur_img = img[:, -1, ...]
        cur_img_metas = [each[len_queue-1] for each in img_metas]

        cur_img_feats = self.extract_feat(img=cur_img, img_metas=cur_img_metas)[0]  

        bbox_list = [dict() for i in range(len(img_metas))]
        metric_dict = self.simple_test_pts(
            cur_img_feats,
            cur_img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            fut_valid_flag=fut_valid_flag,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )
        
        for result_dict in bbox_list:
            result_dict['metric_results'] = metric_dict

        return bbox_list
    
    def simple_test_pts(
        self,
        img_feats,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        fut_valid_flag=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """Test function"""
        B = ego_his_trajs.size(0)
        ego_his_trajs_ = ego_his_trajs.reshape(B, -1)
        ego_lcf_feat_ = ego_lcf_feat.reshape(B, -1)
        ego_fut_cmd_ = ego_fut_cmd.reshape(B, -1)
        ego_info = torch.cat([ego_his_trajs_, ego_lcf_feat_, ego_fut_cmd_], dim=1)

        preds_ego_future_traj, _, _ = self.pts_bbox_head(
                                        img_feats, 
                                        img_metas, 
                                    )

        with torch.no_grad():
            # pre-process
            gt_bbox = gt_bboxes_3d[0][0]
            gt_label = gt_labels_3d[0][0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])

            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = preds_ego_future_traj[0]
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            
            ego_fut_preds = ego_fut_preds.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_preds[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )

            #mid print
            # update metrics
            self.metrics_history.append(metric_dict_planner_stp3)
            self.call_count += 1

            # print results
            if self.call_count % 500 == 0:
                self.compute_and_print_metrics_average()

        return metric_dict_planner_stp3
    
    def compute_and_print_metrics_average(self):
        # compute avg
        avg_metrics = {key: sum(m[key] for m in self.metrics_history) / len(self.metrics_history) for key in self.metrics_history[0]}
        # print avg
        print(f"\n Average metrics after {len(self.metrics_history)} calls: {avg_metrics}")


