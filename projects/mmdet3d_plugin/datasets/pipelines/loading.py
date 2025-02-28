import os, cv2
from typing import Any, Dict, Tuple

import mmcv
import torch
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from PIL import Image


from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

def load_augmented_point_cloud(path, virtual=False, reduce_beams=32):
    # NOTE: following Tianwei's implementation, it is hard coded for nuScenes
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    # NOTE: path definition different from Tianwei's implementation.
    tokens = path.split("/")
    vp_dir = "_VIRTUAL" if reduce_beams == 32 else f"_VIRTUAL_{reduce_beams}BEAMS"
    seg_path = os.path.join(
        *tokens[:-3],
        "virtual_points",
        tokens[-3],
        tokens[-2] + vp_dir,
        tokens[-1] + ".pkl.npy",
    )
    assert os.path.exists(seg_path)
    data_dict = np.load(seg_path, allow_pickle=True).item()

    virtual_points1 = data_dict["real_points"]
    # NOTE: add zero reflectance to virtual points instead of removing them from real points
    virtual_points2 = np.concatenate(
        [
            data_dict["virtual_points"][:, :3],
            np.zeros([data_dict["virtual_points"].shape[0], 1]),
            data_dict["virtual_points"][:, 3:],
        ],
        axis=-1,
    )

    points = np.concatenate(
        [
            points,
            np.ones([points.shape[0], virtual_points1.shape[1] - points.shape[1] + 1]),
        ],
        axis=1,
    )
    virtual_points1 = np.concatenate(
        [virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1
    )
    # note: this part is different from Tianwei's implementation, we don't have duplicate foreground real points.
    if len(data_dict["real_points_indice"]) > 0:
        points[data_dict["real_points_indice"]] = virtual_points1
    if virtual:
        virtual_points2 = np.concatenate(
            [virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1
        )
        points = np.concatenate([points, virtual_points2], axis=0).astype(np.float32)
    return points


def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    # print(pts.size())
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    # [-pi/2, pi/2]
    theta = torch.asin(sine_theta)
    phi = torch.atan2(pts[:, 1], pts[:, 0])

    top_ang = 0.1862
    down_ang = -0.5353

    beam_range = torch.zeros(32)
    beam_range[0] = top_ang
    beam_range[31] = down_ang

    for i in range(1, 31):
        beam_range[i] = beam_range[i - 1] - 0.023275
    # beam_range = [1, 0.18, 0.15, 0.13, 0.11, 0.085, 0.065, 0.03, 0.01, -0.01, -0.03, -0.055, -0.08, -0.105, -0.13, -0.155, -0.18, -0.205, -0.228, -0.251, -0.275,
    #                -0.295, -0.32, -0.34, -0.36, -0.38, -0.40, -0.425, -0.45, -0.47, -0.49, -0.52, -0.54]

    num_pts, _ = pts.size()
    mask = torch.zeros(num_pts)
    if reduce_beams_to == 16:
        for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 4:
        for id in [7, 9, 11, 13]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    # [?] pick the 14th beam
    elif reduce_beams_to == 1:
        chosen_beam_id = 9
        mask = (theta < (beam_range[chosen_beam_id - 1] - 0.012)) * (
            theta > (beam_range[chosen_beam_id] - 0.012)
        )
    else:
        raise NotImplementedError
    # points = copy.copy(pts)
    points = pts[mask]
    # print(points.size())
    return points.numpy()

@PIPELINES.register_module()
class CustomLoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"



@PIPELINES.register_module()
class CustomLoadPointsFromFile:
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["pts_filename"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # TODO: make it more general
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results

@PIPELINES.register_module()
class LoadFrontImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                to_float32=False, 
                color_type='unchanged', 
                load_semantic=False, 
                semantic_path='data/nuscenes/nusc_semantic/front_camera',
                load_flow=False,
                flow_path='data/nuscenes/nusc_optical_flow',
                ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.load_semantic = load_semantic
        self.semantic_path = semantic_path
        # Define the color map for the 30 classes
        self.color_map =  [
            # flat
            (128, 64, 128),  # road
            (244, 35, 232),  # sidewalk
            (250, 170, 160), # parking
            (230, 150, 140), # rail track

            # human
            (220, 20, 60),   # person
            (255, 0, 0),     # rider

            # vehicle
            (0, 0, 142),     # car
            (0, 0, 70),      # truck
            (0, 60, 100),    # bus
            (0, 80, 100),    # on rails
            (0, 0, 230),     # motorcycle
            (119, 11, 32),   # bicycle
            (0, 0, 110),     # caravan
            (0, 0, 90),      # trailer

            # construction
            (70, 70, 70),    # building
            (102, 102, 156), # wall
            (190, 153, 153), # fence
            (180, 165, 180), # guard rail
            (150, 100, 100), # bridge
            (150, 120, 90),  # tunnel

            # object
            (153, 153, 153), # pole
            (153, 153, 153), # pole group
            (250, 170, 30),  # traffic sign
            (220, 220, 0),   # traffic light

            # nature
            (107, 142, 35),  # vegetation
            (152, 251, 152), # terrain

            # sky
            (70, 130, 180),  # sky

            # void
            (81, 0, 81),     # ground
            (111, 74, 0),    # dynamic
            (81, 0, 81),     # static
        ]

        self.load_flow = load_flow
        self.flow_path = flow_path


    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # front image only
        front_img_filename = filename[0]
        filename = [front_img_filename]

        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        
        # semantics
        if self.load_semantic:
            semantic_filename = os.path.join(self.semantic_path, front_img_filename.split('/')[-1])
            semantic_img = cv2.imread(semantic_filename, cv2.IMREAD_UNCHANGED)
            colored_semantic_img = self.semantic_to_colored(semantic_img)
            results['semantic_img'] = [colored_semantic_img]
        
        if self.load_flow:
            # hardcode 2 flow now.
            flow_filename_0 = os.path.join(self.flow_path, front_img_filename.split('/')[-1].replace('.jpg', ''), 'flow_0.png')
            flow_filename_1 = os.path.join(self.flow_path, front_img_filename.split('/')[-1].replace('.jpg', ''), 'flow_1.png')

            if os.path.isfile(flow_filename_0):
                flow_0 = self.image_to_flow(flow_filename_0)
                flow_1 = self.image_to_flow(flow_filename_1)
            else:
                flow_0 = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
                flow_1 = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)

            results['flow_img'] = [flow_0, flow_1]

        return results
    
    def image_to_flow(self, image_path, scale_factor=65.536):
        # 加载16位深度的图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # 将PIL图像转换为Numpy数组
        img_array = np.array(image, dtype='uint16')

        # 提取u和v分量
        u = img_array[..., 1].astype(np.float32)
        v = img_array[..., 2].astype(np.float32)

        # 逆映射到原始光流数据范围
        u = (u - 32768) / scale_factor
        v = (v - 32768) / scale_factor

        # 堆叠回光流格式
        flow = np.dstack((u, v))

        return flow

    def semantic_to_colored(self, semantic_img):
        # Create an empty RGB image with the same height and width as the semantic image
        colored_semantic_img = np.zeros((semantic_img.shape[0], semantic_img.shape[1], 3), dtype=np.float32)

        # Populate the RGB image using the color map
        for class_value in range(len(self.color_map)):
            mask = semantic_img == class_value
            colored_semantic_img[mask] = np.array(self.color_map[class_value], dtype=np.float32)

        return colored_semantic_img

@PIPELINES.register_module()
class LoadSingleViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                to_float32=False, 
                color_type='unchanged', 
                load_semantic=False, 
                semantic_path='data/nuscenes/nusc_semantic/front_camera',
                load_flow=False,
                flow_path='data/nuscenes/nusc_optical_flow',
                view_name='front',
                ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.load_semantic = load_semantic
        self.semantic_path = semantic_path
        self.view_name = view_name
        self.view_idx_mapping = {
            'front': 0,
            'front_right': 1,
            'front_left': 2,
            'back': 3,
            'back_left': 4,
            'back_right': 5,
        }
        # Define the color map for the 30 classes
        self.color_map =  [
            # flat
            (128, 64, 128),  # road
            (244, 35, 232),  # sidewalk
            (250, 170, 160), # parking
            (230, 150, 140), # rail track

            # human
            (220, 20, 60),   # person
            (255, 0, 0),     # rider

            # vehicle
            (0, 0, 142),     # car
            (0, 0, 70),      # truck
            (0, 60, 100),    # bus
            (0, 80, 100),    # on rails
            (0, 0, 230),     # motorcycle
            (119, 11, 32),   # bicycle
            (0, 0, 110),     # caravan
            (0, 0, 90),      # trailer

            # construction
            (70, 70, 70),    # building
            (102, 102, 156), # wall
            (190, 153, 153), # fence
            (180, 165, 180), # guard rail
            (150, 100, 100), # bridge
            (150, 120, 90),  # tunnel

            # object
            (153, 153, 153), # pole
            (153, 153, 153), # pole group
            (250, 170, 30),  # traffic sign
            (220, 220, 0),   # traffic light

            # nature
            (107, 142, 35),  # vegetation
            (152, 251, 152), # terrain

            # sky
            (70, 130, 180),  # sky

            # void
            (81, 0, 81),     # ground
            (111, 74, 0),    # dynamic
            (81, 0, 81),     # static
        ]

        self.load_flow = load_flow
        self.flow_path = flow_path



    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # front image only
        view_idx = self.view_idx_mapping[self.view_name]
        single_img_filename = filename[view_idx]
        assert self.view_name.upper()+'/' in single_img_filename
        filename = [single_img_filename]

        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        
        # semantics
        if self.load_semantic:
            semantic_filename = os.path.join(self.semantic_path, front_img_filename.split('/')[-1])
            semantic_img = cv2.imread(semantic_filename, cv2.IMREAD_UNCHANGED)
            colored_semantic_img = self.semantic_to_colored(semantic_img)
            results['semantic_img'] = [colored_semantic_img]
        
        if self.load_flow:
            # hardcode 2 flow now.
            flow_filename_0 = os.path.join(self.flow_path, front_img_filename.split('/')[-1].replace('.jpg', ''), 'flow_0.png')
            flow_filename_1 = os.path.join(self.flow_path, front_img_filename.split('/')[-1].replace('.jpg', ''), 'flow_1.png')

            if os.path.isfile(flow_filename_0):
                flow_0 = self.image_to_flow(flow_filename_0)
                flow_1 = self.image_to_flow(flow_filename_1)
            else:
                flow_0 = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
                flow_1 = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)

            results['flow_img'] = [flow_0, flow_1]

        return results