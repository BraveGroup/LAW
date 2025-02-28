import torch, cv2
import numpy as np

def prj_ego_traj_to_2d(ego_fut_trajs, img_metas):
    lidar2img_mat = torch.from_numpy(np.stack(img_metas[0]['lidar2img'], axis=0)).to(ego_fut_trajs.device).float()
    img_shape = img_metas[0]['img_shape'][0]
    H, W = img_shape[0], img_shape[1]

    # 1. Convert 2D future trajectories to 3D
    zeros_column = torch.zeros_like(ego_fut_trajs[..., :1]) - img_metas[0]['lidar2ego_translation'][-1]
    ego_fut_trajs_3d = torch.cat((ego_fut_trajs, zeros_column), dim=-1)
    #My TODO: use the front ego ground point
    ego_fut_trajs_3d[..., 1] += 4.75

    # 2. Project the 3D points to the image
    ref_pts_cam, pts_mask = prj_pts_to_img(ego_fut_trajs_3d, lidar2img_mat[0:1], H, W) 
    return ref_pts_cam, pts_mask

def prj_pts_to_img(pts, lidar2img_list, H, W):
    """
    pts: N, 2
    lidar2img_list: K, 3, 4
    """
    lidar2img = torch.tensor(lidar2img_list).unsqueeze(1)
    reference_points_raw = pts
    num_pts = reference_points_raw.shape[0]
    num_cam = lidar2img.shape[0]
    reference_points = torch.cat([reference_points_raw, torch.ones_like(reference_points_raw[..., 0:1])], dim=-1)

    lidar2img = lidar2img.repeat(1, num_pts, 1, 1)
    reference_points = reference_points.unsqueeze(-1).unsqueeze(0).repeat(num_cam, 1, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-2
    pts_mask_raw = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    ref_pt_cam = reference_points_cam.clone()

    for idx in range(num_cam):
        reference_points_cam[idx, :, 0] /= W
        reference_points_cam[idx, :, 1] /= H
    reference_points_cam = (reference_points_cam - 0.5) * 2
    
    pts_mask = (pts_mask_raw & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    return ref_pt_cam, pts_mask.squeeze(-1)

def draw_lidar_pts(ref_pts_cam, pts_mask, img, length = 6, color = (255, 255, 0)):
    ref_pts_real = ref_pts_cam.clone()
    num_cam = pts_mask.shape[0]

    ref_single = ref_pts_real[0]
    pts_mask_single = pts_mask[0]
    ref_this_cam = ref_single[pts_mask_single]
    pts = ref_this_cam.detach().cpu().int().numpy()
    for pt in pts:
        cv2.circle(img, (pt[0], pt[1]), length, color, -1)
    return img

def denormalize_img(norm_img, img_metas):
    device = norm_img.device
    # 获取均值和标准差
    mean = torch.tensor(img_metas[0]['img_norm_cfg']['mean'], dtype=torch.float32).to(device)
    std = torch.tensor(img_metas[0]['img_norm_cfg']['std'], dtype=torch.float32).to(device)
    
    # 拓展维度以匹配图像的维度
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    
    # 对标准化的图像进行反标准化
    img = norm_img * std + mean
    
    return img

