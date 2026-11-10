#!/usr/bin/python3
# coding=utf8
# @Author: Aiden
# @Date: 2024/12/31
import cv2
import copy
import math
import numpy as np
from sdk import common
from typing import Tuple, List, Optional

# Physical parameters of the robotic gripper with units in meters. 机械臂夹持器的物理参数(单位:米)
GRIPPER_HB = 0.014  
GRIPPER_BC = 0.03 
GRIPPER_ED = 0.037
GRIPPER_DC = 0.022
EDC = math.radians(180 - 21)
GRIPPER_IH = 0.02
GRIPPER_IG = 0.005
LCD = math.acos((GRIPPER_HB - GRIPPER_IG) / GRIPPER_IH)
GRIPPER_EC = (GRIPPER_ED**2 + GRIPPER_DC**2 - 2 * GRIPPER_ED * GRIPPER_DC * math.cos(EDC)) ** 0.5
ECD = math.acos((GRIPPER_DC**2 + GRIPPER_EC**2 - GRIPPER_ED**2) / (2 * GRIPPER_DC * GRIPPER_EC))

# 机械臂默认舵机脉宽（与 color_sorting/object_sorting 等默认姿态一致）；标定节点与 calibration GUI 依赖
joint_angle = [500, 520, 185, 50, 500]

# 工作区顶面相对标定平面的高度 (m)，RGB 黄框与深度 ROI 共用
WORK_AREA_PLANE_HEIGHT = 0.035

def get_gripper_size(angle: float, angle_zero=200) -> Tuple[float, float]:
    """
    Compute the gripper's width and height based on its opening angle. 根据夹持器角度计算夹持器的宽度和高度
    
    Args:
        angle: Gripper opening angle with a range from 0 to 1000. 夹持器角度(0-1000)
        angle_zero: Zero reference angle for the gripper, set to 200 by default). 夹持器角度零点(默认200)
    Returns:
        width: Gripper width 夹持器宽度
        height: Gripper height 夹持器高度
    """
    angle = math.radians((angle - angle_zero) / 1000 * 180)
     
    GRIPPER_BJ = math.cos(angle) * GRIPPER_BC
    GRIPPER_HJ = GRIPPER_HB + GRIPPER_BJ
    GRIPPER_LC = math.cos(LCD + ECD) * GRIPPER_EC
    GRIPPER_KE = GRIPPER_HJ - GRIPPER_LC

    GRIPPER_KE = GRIPPER_HB + math.cos(angle) * GRIPPER_BC - math.cos(LCD + ECD) * GRIPPER_EC

    GRIPPER_JC = (GRIPPER_BC**2 - GRIPPER_BJ**2)**0.5
    GRIPPER_LE = (GRIPPER_EC**2 - GRIPPER_LC**2)**0.5
    gripper_depth = GRIPPER_JC + GRIPPER_LE
    gripper_width = 2*GRIPPER_KE
    
    return gripper_width, gripper_depth

def set_gripper_size(width: float) -> int:
    """
    Compute the gripper angle required for a target width. 根据目标宽度计算夹持器需要的角度
    
    Args:
        width: Target width in meters. 目标宽度(米)
        
    Returns:
        angle: Gripper angle (range: 0–1000). 夹持器角度(0-1000)
    """
    width = width / 2
    a = (width - GRIPPER_HB + math.cos(LCD + ECD) * GRIPPER_EC) / GRIPPER_BC
    a = max(-1.0, min(1.0, a))
    return int(math.degrees(math.acos(a)) / 180 * 1000 + 200)

def world_to_pixels(world_points, K, T):
    """
    Convert world coordinates to pixel coordinates. 将世界坐标点转换为像素坐标
    Args:
        world_points: List of world coordinates. 世界坐标点列表
        K: Camera intrinsic matrix 相机内参矩阵
        T: Extrinsic transformation matrix [R|t] 外参矩阵 [R|t]
    Returns:
        pixel_points: List of corresponding pixel coordinates 像素坐标点列表
    """
    pixel_points = []
    for wp in world_points:
        # Convert world coordinates to homogeneous coordinates. 将世界坐标转换为齐次坐标
        world_homo = np.append(wp, 1).reshape(4, 1)
        # Transform to camera coordinates using the extrinsic matrix. 通过外参矩阵转换到相机坐标系
        camera_point = np.dot(T, world_homo)
        # Project onto the image plane using the intrinsic matrix. 投影到像素平面
        pixel_homo = np.dot(K, camera_point[:3])
        # Normalization 归一化
        pixel = (pixel_homo / pixel_homo[2])[:2].reshape(-1)
        pixel_points.append(pixel)
    return pixel_points

def calculate_pixel_length(world_length, K, T):
    """
    Compute the corresponding pixel length for a given length in world coordinates. 计算世界坐标中的长度在像素坐标中的对应长度
    Args:
        world_length: Length in world space 世界坐标中的长度
        K: Camera intrinsic matrix 相机内参矩阵
        T: Extrinsic transformation matrix 外参矩阵
    Returns:
        pixel_length: Corresponding length in pixel space 像素坐标中的长度
    """
    # Define a starting point and direction. 定义起始点和方向
    start_point = np.array([0, 0, 0])  # starting point 起始点
    direction = np.array([0, 1, 0])  # y-direction y方向

    # Compute the endpoint. 计算终点坐标
    end_point = start_point + direction * world_length
    # Transform both endpoints to pixel coordinates. 转换两个端点到像素坐标
    pixels = world_to_pixels([start_point, end_point], K, T)
    # Calculate Euclidean distance in pixel space. 计算像素距离
    pixel_length = np.linalg.norm(pixels[1] - pixels[0])

    return int(pixel_length)

def get_plane_values(depth_image: np.ndarray, 
                    plane: Tuple[float, float, float, float],
                    intrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the distance from each depth pixel to a given plane. 计算深度图像中每个点到平面的距离
    
    Args:
        depth_image: Depth image 深度图像
        plane: Plane equation parameters 平面方程参数(a,b,c,d)
        intrinsic_matrix: Camera intrinsic matrix 相机内参矩阵
        
    Returns:
        plane_values: Distance from each pixel to the plane 每个点到平面的距离
    """
    a, b, c, d = plane
    # Extract camera intrinsics 提取相机内参
    fx = intrinsic_matrix[0]
    fy = intrinsic_matrix[4]
    cx = intrinsic_matrix[2]
    cy = intrinsic_matrix[5]
    
    # Image dimensions 图像尺寸
    H, W = depth_image.shape
    
    # Generate pixel coordinate grid 生成像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Calculate the camera coordinates 计算相机坐标
    z = depth_image / 1000.0  # Convert units to meters 转换为米
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Evaluate the plane equation values 计算平面方程值
    plane_values = a * x + b * y + c * z + d
    
    return plane_values


def _camera_k_d(camera_info, intrinsic=None, distortion=None):
    if intrinsic is not None:
        K = np.matrix(intrinsic).reshape(1, -1, 3) if not isinstance(intrinsic, np.matrix) else intrinsic
    else:
        K = np.matrix(camera_info.k).reshape(1, -1, 3)
    if distortion is not None:
        D = np.array(distortion)
    else:
        D = np.array(camera_info.d)
    return K, D


def project_work_area_corners(
    corners: np.ndarray,
    extrinsic: np.ndarray,
    camera_info: object = None,
    intrinsic=None,
    distortion=None,
    plane_height: float = WORK_AREA_PLANE_HEIGHT,
) -> np.ndarray:
    """将工作区四角投影到图像像素（与 RGB 黄框同一套外参与高度）。"""
    translation_vec = extrinsic[:1]
    rotation_mat = extrinsic[1:]
    corners_copy = np.asarray(corners).reshape(-1, 3)
    tvec, rmat = common.extristric_plane_shift(
        np.array(translation_vec).reshape((3, 1)),
        np.array(rotation_mat),
        plane_height,
    )
    K, D = _camera_k_d(camera_info, intrinsic, distortion)
    imgpts, _ = cv2.projectPoints(corners_copy[:-1], np.array(rmat), np.array(tvec), K, D)
    return np.int32(imgpts).reshape(-1, 2)


def remap_points_rgb_to_depth(
    rgb_pts: np.ndarray,
    rgb_camera_info: object,
    depth_camera_info: object,
) -> np.ndarray:
    """按分辨率将 RGB 像素坐标缩放到深度图坐标系。"""
    rgb_pts = np.asarray(rgb_pts, dtype=np.float64)
    sx = depth_camera_info.width / max(int(rgb_camera_info.width), 1)
    sy = depth_camera_info.height / max(int(rgb_camera_info.height), 1)
    out = rgb_pts.copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    out = np.int32(np.round(out))
    out[:, 0] = np.clip(out[:, 0], 0, int(depth_camera_info.width) - 1)
    out[:, 1] = np.clip(out[:, 1], 0, int(depth_camera_info.height) - 1)
    return out


def create_roi_mask(
    depth_image: np.ndarray,
    bgr_image: np.ndarray,
    corners: np.ndarray,
    camera_info: object,
    extrinsic: np.ndarray,
    max_height: float,
    max_obj_height: float = WORK_AREA_PLANE_HEIGHT,
    depth_camera_info: object = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用与 RGB 黄框一致的四边形遮罩裁剪深度图。
    Returns:
        depth_image: 区域外置为 max_height 的深度图
        roi_pts: 深度图坐标系下的 ROI 四角（用于绘制）
    """
    image_height, image_width = depth_image.shape[:2]
    rgb_pts = project_work_area_corners(
        corners, extrinsic, camera_info, plane_height=max_obj_height,
    )
    if depth_camera_info is not None and (
        depth_camera_info.width != camera_info.width
        or depth_camera_info.height != camera_info.height
    ):
        roi_pts = remap_points_rgb_to_depth(rgb_pts, camera_info, depth_camera_info)
    else:
        roi_pts = rgb_pts.copy()

    mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_pts], 255)
    depth_out = depth_image.copy()
    depth_out[mask == 0] = max_height
    return depth_out, roi_pts

def find_depth_range(depth_image: np.ndarray, max_distance: float) -> Tuple[float, float]:
    """
    Find the minimum distance in a depth image. 查找深度图像中的最小
    Args:
        depth_image: Depth image 深度图像
    Returns:
        min_distance: Minimum distance in millimeters. 最小距离(mm)
    """
    height, width = depth_image.shape[:2]
    
    # Process depth data 处理深度数据
    depth = np.copy(depth_image).reshape(-1)
    depth[depth <= 0] = max_distance  # Set invalid values to max_distance 将无效值设为max_distance
    
    # Find the closest point 找到最近点
    min_idx = np.argmin(depth)
    min_y, min_x = min_idx // width, min_idx % width
    min_distance = depth_image[min_y, min_x] 
    
    return min_distance

def extract_contours(
    plane_values: np.ndarray,
    filter_height: float
) -> List[np.ndarray]:
    """
    Extract contours from a depth image. 提取深度图像中的轮廓
    Args:
        plane_values: Plane value 平面值
        filter_height: Height threshold for filtering 过滤高度
    Returns:
        contours: List of extracted contours 轮廓列表
    """
    # Apply height threshold 过滤高度
    filtered_image = np.where(plane_values <= filter_height, 0, 255).astype(np.uint8)
    
    # Perform binarization and contour extraction 二值化和轮廓提取
    _, binary = cv2.threshold(filtered_image, 1, 255, cv2.THRESH_BINARY)
    # cv2.imshow(color, binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours

def convert_depth_to_camera_coords(
    pixel_coords: Tuple[float, float, float],
    intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """
    Convert depth pixel coordinates to camera coordinate system. 将深度像素坐标转换为相机坐标系
    Args:
        pixel_coords: Pixel coordinates 像素坐标 (x, y, z)
        intrinsic_matrix: Camera intrinsic matrix 相机内参矩阵
    Returns:
        camera_coords: Coordinates in the camera coordinate system 相机坐标系下的坐标
    """
    fx, fy = intrinsic_matrix[0], intrinsic_matrix[4]
    cx, cy = intrinsic_matrix[2], intrinsic_matrix[5]
    px, py, pz = pixel_coords
    
    x = (px - cx) * pz / fx
    y = (py - cy) * pz / fy
    z = pz
    
    return np.array([x, y, z])

def calculate_world_position(
    pixel_x: float,
    pixel_y: float,
    depth: float,
    plane: Tuple[float, float, float, float],
    endpoint: np.ndarray,
    hand2cam_tf_matrix: np.ndarray,
    intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """
    深度像素 → 机械臂基坐标系世界坐标（与 calibration 手眼链一致：endpoint @ hand2cam @ T_cam）。
    plane 保留入参以兼容旧调用，Z 由齐次变换给出；抓取节点可再覆盖 Z。
    """
    del plane  # 未使用；避免误用相机系平面方程覆盖世界 Z
    camera_position = convert_depth_to_camera_coords(
        [pixel_x, pixel_y, depth / 1000.0],
        intrinsic_matrix,
    )
    pose_cam = common.xyz_euler_to_mat(camera_position, (0, 0, 0))
    pose_end = np.matmul(hand2cam_tf_matrix, pose_cam)
    pose_world = np.matmul(endpoint, pose_end)
    return np.array(pose_world[:3, 3], dtype=np.float64)


