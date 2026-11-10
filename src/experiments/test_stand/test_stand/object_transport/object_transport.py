#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/11/18
import os
import shutil
import cv2
import math
import yaml
import copy
import time
import torch
import queue
import rclpy
import threading
import numpy as np
import sdk.fps as fps
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, SetBool, Empty
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from tf2_ros import Buffer, TransformListener, TransformException
from ultralytics.models.fastsam import FastSAMPredictor
from ultralytics.utils.ops import scale_masks
from app.common import Heart
from large_models_msgs.srv import SetBox
from kinematics_msgs.srv import SetRobotPose, SetJointValue, GetRobotPose
from kinematics.kinematics_control import set_joint_value_target
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition
from test_stand.color_sorting.utils import utils, calculate_grasp_yaw, pick_and_place, common 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def prompt(results, bboxes=None, points=None, labels=None, texts=None, log=None):
    if bboxes is None and points is None and texts is None:
        return results
    prompt_results = []
    if not isinstance(results, list):
        results = [results]
    for result in results:
        if len(result) == 0:
            prompt_results.append(result)
            continue
        #强制 Mask 掩码和原始图片一样
        masks = result.masks.data
        if masks.shape[1:] != result.orig_shape:
            masks = scale_masks(masks[None], result.orig_shape)[0]
        idx = torch.zeros(len(result), dtype=torch.bool, device=device)
        if bboxes is not None:
            bboxes = torch.as_tensor(
                np.asarray(bboxes, dtype=np.int32), dtype=torch.int32, device=device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            mask_areas = torch.stack([masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in bboxes])
            full_mask_areas = torch.sum(masks, dim=(1, 2))
 
            u = mask_areas / full_mask_areas  
            u = torch.nan_to_num(u, nan=0.0) 
            indices = (u >= (torch.max(u) - 0.1)).nonzero(as_tuple=True)[1] 
            u1 = full_mask_areas / bbox_areas
            max_index = indices[torch.argmax(u1[indices])]
            idx[max_index] = True

        prompt_results.append(result[idx])

    return prompt_results


# calibration.CalibrationNode 未乘 TF 前手眼常量，与 calibration.py 保持一致
_HAND2CAM_BASE_DEFAULT = np.array(
    [
        [0.0, 0.0, 1.0, -0.101],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.05],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class ObjectTransport(Node):
    # 显示裁剪：裁剪顶部和底部区域
    DISPLAY_CROP_TOP = 40
    DISPLAY_CROP_END = 440

    # 缓存机械臂当前位姿的时间（秒）
    ENDPOINT_CACHE_SEC = 45.0

    # 放置补偿：顶面之上间隙（米）；实际落点还受 PLACE_Z_BIAS 微调
    PLACE_SURFACE_CLEARANCE = 0.003  # 放置：目标顶面之上间隙
    PLACE_Z_BIAS = -0.06  # 放置：Z 微调
    RESTORE_PICK_X_BIAS = 0.00  # 复原夹取：世界坐标 X 微调（米）

    # 深度 Z：plane_values + 工作面 + DEPTH_Z_COMPENSATION
    DEPTH_Z_COMPENSATION = -0.00  # 深度 Z 补偿（米）
    PICK_GRIPPER_DEPTH = 0.06  # 夹取：夹取深度（米）
    PICK_APPROACH_OFFSET_Z = 0.000  # 夹取：夹取偏移（米）

    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self._ensure_param('use_depth', True)
        self._ensure_param('depth_fuse_rgb_xy', True)
        self._ensure_param('display', True)
        self._ensure_param('start', False)
        self._ensure_param('rgb_image_topic', '/depth_cam/rgb0/image_raw')
        self._ensure_param('rgb_camera_info_topic', '/depth_cam/rgb0/camera_info')
        self._ensure_param('depth_image_topic', '/depth_cam/depth0/image_raw')
        self._ensure_param('depth_camera_info_topic', '/depth_cam/depth0/camera_info')
        self._ensure_param('tf_camera_mount_frame', 'camera_link0')
        self._ensure_param('tf_rgb_frame', 'rgb_camera_link')
        # RGB 与 agent_process 一致用 RELIABLE；深度图用 sensor_data（BEST_EFFORT）
        self._rgb_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._sensor_qos = qos_profile_sensor_data
        self._sensor_cb_group = ReentrantCallbackGroup()
        self._image_received = False
        self.fps = fps.FPS() # 帧率统计器(frame rate counter)
        self.image_queue = queue.Queue(maxsize=2)
        self.depth_image_queue = queue.Queue(maxsize=2)
        self._calibration_config_dir = self._resolve_calibration_config_dir()
        self._init_parameters()
        self.hand2cam_tf_matrix = np.array(_HAND2CAM_BASE_DEFAULT, copy=True)
        self.endpoint = None
        self._endpoint_updated_at = 0.0
        self.corners = None
        self.plane = None
        self.depth_ready = False

        self.set_above = False
        self.record_position = []
        self.lock = threading.RLock()
        self.bridge = CvBridge()
        self.gripper_pixel_size = utils.get_gripper_size(500)

        self.get_logger().info("Initializing publishers...")
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
        self.transport_finished_pub = self.create_publisher(Bool, '~/transport_finished', 1)
        
        self.get_logger().info("Initializing callback groups and clients...")
        timer_cb_group = ReentrantCallbackGroup()
        self.set_joint_value_target_client = self.create_client(SetJointValue, 'kinematics/set_joint_value_target', callback_group=timer_cb_group)
        self.set_joint_value_target_client.wait_for_service()

        
        self.kinematics_client = self.create_client(
            SetRobotPose, 'kinematics/set_pose_target', callback_group=timer_cb_group)
        self.kinematics_client.wait_for_service()

        self.get_current_pose_client = self.create_client(
            GetRobotPose,
            '/kinematics/get_current_pose',
            callback_group=timer_cb_group,
        )
        self.get_current_pose_client.wait_for_service()

        self.enter_srv = self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.exit_srv = self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.enable_transport_srv = self.create_service(SetBool, '~/enable_transport', self.enable_transport_srv_callback)
        self.set_pick_position_srv = self.create_service(SetBox, '~/set_pick_position', self.set_pick_position_srv_callback)
        self.set_place_position_srv = self.create_service(SetBox, '~/set_place_position', self.set_place_position_srv_callback)
        self.record_position_srv = self.create_service(SetBox, '~/record_position', self.record_position_srv_callback)
        self.clear_record_position_srv = self.create_service(
            Empty, '~/clear_record_position', self.clear_record_position_srv_callback)
        self.abort_srv = self.create_service(Trigger, '~/abort', self.abort_srv_callback)

        self.get_logger().info("Setting up FastSAM model...")
        fastsam_model = self._resolve_fastsam_model_path()
        self.get_logger().info(f'FastSAM model: {fastsam_model}')
        overrides = dict(
            conf=0.4,
            task='segment',
            mode='predict',
            model=fastsam_model,
            save=False,
            imgsz=(640),
        )

        self.predictor = FastSAMPredictor(overrides=overrides)

        self.get_logger().info("Warming up FastSAM model...")
        self.predictor(np.zeros((640, 480, 3), dtype=np.uint8))
        self.get_logger().info("...FastSAM model warmed up.")

        self.get_logger().info(
            f'标定参数: {self._calibration_config_dir}/transform.yaml, calibration.yaml')

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    @staticmethod
    def _resolve_calibration_config_dir():
        try:
            return os.path.join(get_package_share_directory('calibration'), 'config')
        except Exception:
            return '/home/ubuntu/ros2_ws/src/calibration/config'

    def _load_calibration_params(self):
        path = os.path.join(self._calibration_config_dir, 'calibration.yaml')
        return common.get_yaml_data(path)

    def _init_hand2cam_tf(self):
        tf_mount = self.get_parameter('tf_camera_mount_frame').value
        tf_rgb = self.get_parameter('tf_rgb_frame').value
        tf_buffer = Buffer()
        TransformListener(tf_buffer, self)
        deadline = time.time() + 30.0
        transform = None
        while rclpy.ok() and time.time() < deadline:
            try:
                if tf_buffer.can_transform(
                        tf_rgb, tf_mount, rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.5)):
                    transform = tf_buffer.lookup_transform(
                        tf_rgb, tf_mount, rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0))
                    break
            except TransformException:
                pass

            time.sleep(0.05)
        if transform is None:
            self.get_logger().error(
                f'手眼 TF 超时: {tf_mount} -> {tf_rgb}，深度定位可能不准')
            return
        t = transform.transform.translation
        r = transform.transform.rotation
        mount_tf = common.xyz_quat_to_mat(
            [t.x, t.y, t.z], [r.w, r.x, r.y, r.z])
        self.hand2cam_tf_matrix = np.matmul(mount_tf, self.hand2cam_tf_matrix)
        self.get_logger().info('手眼变换矩阵已更新（depth 模式）')

    def _refresh_robot_endpoint(self, timeout_sec=8.0, force=False):
        if (not force and self.endpoint is not None
                and time.time() - self._endpoint_updated_at < self.ENDPOINT_CACHE_SEC):
            return True
        try:
            res = self.send_request(
                self.get_current_pose_client, GetRobotPose.Request(), timeout_sec=timeout_sec)
        except TimeoutError:
            self.get_logger().warning(
                '获取机械臂当前位姿超时，将跳过深度定位（避免使用过期 endpoint）')
            return False
        pose_t = res.pose.position
        pose_r = res.pose.orientation
        self.endpoint = common.xyz_quat_to_mat(
            [pose_t.x, pose_t.y, pose_t.z],
            [pose_r.w, pose_r.x, pose_r.y, pose_r.z],
        )
        self._endpoint_updated_at = time.time()
        return True

    @staticmethod
    def _is_world_position_plausible(position) -> bool:
        try:
            x, y, z = float(position[0]), float(position[1]), float(position[2])
        except (TypeError, IndexError):
            return False
        if x < 0.06 or x > 0.42:
            return False
        if abs(y) > 0.22:
            return False
        if z < -0.05 or z > 0.45:
            return False
        return True

    @staticmethod
    def _action_vllm_box(box_info):
        if not box_info:
            return None
        if box_info[0] in ('pick', 'restore_pick') and len(box_info) >= 2:
            return box_info[1]
        if box_info[0] == 'place' and len(box_info) >= 3:
            return box_info[-1]
        if box_info[0] == 'record' and len(box_info) >= 3:
            return box_info[-1]
        return None

    def _draw_vllm_box(self, bgr_image, box_info):
        box = self._action_vllm_box(box_info)
        if box and len(box) >= 4:
            cv2.rectangle(
                bgr_image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0), 2, 1)

    def _ensure_param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def get_node_state(self, request, response):
        return response

    def _resolve_fastsam_model_path(self):
        candidates = [
            os.path.join(
                get_package_share_directory('test_stand'),
                'test_stand', 'resources', 'models', 'FastSAM-x.pt',
            ),
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'resources', 'models', 'FastSAM-x.pt',
            ),
        ]
        min_bytes = 100_000_000  # 完整 FastSAM-x 约 145MB
        for path in candidates:
            if os.path.isfile(path) and os.path.getsize(path) >= min_bytes:
                return path
        sizes = {p: (os.path.getsize(p) if os.path.isfile(p) else -1) for p in candidates}
        raise FileNotFoundError(
            f'FastSAM-x.pt 未找到或文件不完整（需 >= {min_bytes} 字节）。已检查: {sizes}'
        )

    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下左键
            self.mouse_click = True
            self.drag_start = (x, y)  # 记录起始位置
            self.drag_type = 'pick'   # 记录拖动类型
            self.track_window = None
            self.selection = None

        elif event == cv2.EVENT_RBUTTONDOWN: # 按下右键 (用于 Place)
            self.mouse_click = True
            self.drag_start = (x, y)  # 记录起始位置
            self.drag_type = 'place'  # 记录拖动类型
            self.track_window = None
            self.selection = None

        elif event == cv2.EVENT_MBUTTONDOWN: # 按下中键 (用于清空)
            self.mouse_click = False
            self.selection = None
            self.track_window = None
            self.drag_start = None
            self.drag_type = None
        
        if self.drag_start:  # 如果正在拖动
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax) # 实时更新选框
        
        if event == cv2.EVENT_LBUTTONUP:  # 松开左键
            if self.drag_start and self.selection and self.drag_type == 'pick':
                self.track_window = self.selection # 最终选框
                with self.lock:
                    # self.get_logger().info(f'鼠标框选, 添加到抓取队列: {self.track_window}')
                    self.action_list.append(['pick', self.track_window])
            
            self.mouse_click = False
            self.drag_start = None
            self.selection = None # 拖动选框清空
            self.drag_type = None


        elif event == cv2.EVENT_RBUTTONUP:  # 松开右键
            if self.drag_start and self.selection and self.drag_type == 'place':
                self.track_window = self.selection # 最终选框
                with self.lock:
                    default_offset = [0.0, 0.0]
                    self.action_list.append(['place', default_offset,self.track_window])
            #重置状态
            self.mouse_click = False
            self.selection = None
            self.track_window = None
            self.drag_start = None

    def _init_parameters(self):
        self.heart = None
        self.enter = False
        self.start_transport = False
        self.enable_transport = False
        self.start_get_roi = False
        self.image_sub = None
        self.camera_info_sub = None
        self.latest_camera_info = None
        self.depth_image_sub = None
        self.depth_cam_info_sub = None
        self.white_area_center = None
        self.extristric = None
        self.intrinsic = None
        self.latest_depth_img = None
        self.latest_depth_cam_info = None
        self.depth_lock = threading.Lock()
        self.action_list = []
        self.target = []
        self.start_stamp = time.time()
        self._endpoint_updated_at = 0.0

        self.mouse_click = False
        self.selection = None
        self.track_window = None
        self.drag_start = None
        self.drag_type = None

    def init_process(self):
        self.timer.cancel()
        # 必须先暴露 init_finish，否则 vllm_object_transport 会一直阻塞
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'object_transport ready')

        def _background_boot():
            if self.get_parameter('use_depth').value:
                self._init_hand2cam_tf()
            threading.Thread(target=self.main, daemon=True).start()
            threading.Thread(target=self.transport_thread, daemon=True).start()
            if self.get_parameter('start').value:
                self.enter_srv_callback(Trigger.Request(), Trigger.Response())
                req = SetBool.Request()
                req.data = True
                self.enable_transport_srv_callback(req, SetBool.Response())

        threading.Thread(target=_background_boot, daemon=True).start()

    def send_request(self, client, msg, timeout_sec=30.0):
        future = client.call_async(msg)
        done = threading.Event()

        def _on_done(_future):
            done.set()

        future.add_done_callback(_on_done)
        if not done.wait(timeout_sec):
            raise TimeoutError(f'service call timeout: {client.srv_name}')
        if future.exception() is not None:
            raise future.exception()
        return future.result()

    def go_home(self, interrupt=True, back=True):
        if interrupt:
            set_servo_position(self.joints_pub, 0.5, ((10, 200), ))
            time.sleep(0.5)

        joint_angle = [500, 520, 185, 50, 500]
        msg = set_joint_value_target(joint_angle)
        try:
            self.send_request(self.set_joint_value_target_client, msg, timeout_sec=60.0)
        except TimeoutError:
            self.get_logger().warning(
                'kinematics/set_joint_value_target 超时，改用舵机指令回零')
        set_servo_position(self.joints_pub, 1, (
            (2, joint_angle[1]), (3, joint_angle[2]), (4, joint_angle[3]), (5, 500)))
        time.sleep(1)

        if back:
            set_servo_position(self.joints_pub, 1, ((1, joint_angle[0]),))
            time.sleep(1.5)

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "enter object transport")
        with self.lock:
            self._init_parameters()
            self.heart = Heart(self, '~/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # Heartbeat package 心跳包
            cv2.namedWindow('image', 1)
            cv2.setMouseCallback('image', self.onmouse)

            if self.image_sub is None:
                rgb_topic = self.get_parameter('rgb_image_topic').value
                info_topic = self.get_parameter('rgb_camera_info_topic').value
                self.get_logger().info(f'订阅相机: {rgb_topic} / {info_topic}')
                self.image_sub = self.create_subscription(
                    Image, rgb_topic, self.image_callback, self._rgb_qos,
                    callback_group=self._sensor_cb_group)
                self.camera_info_sub = self.create_subscription(
                    CameraInfo, info_topic, self.camera_info_callback, self._rgb_qos,
                    callback_group=self._sensor_cb_group)

            if self.get_parameter('use_depth').value and self.depth_image_sub is None:
                depth_topic = self.get_parameter('depth_image_topic').value
                depth_info_topic = self.get_parameter('depth_camera_info_topic').value
                self.get_logger().info(f'订阅深度: {depth_topic} / {depth_info_topic}')
                self.depth_image_sub = self.create_subscription(
                    Image, depth_topic, self.depth_image_callback, self._sensor_qos,
                    callback_group=self._sensor_cb_group)
                self.depth_cam_info_sub = self.create_subscription(
                    CameraInfo, depth_info_topic, self.depth_camera_info_callback,
                    self._sensor_qos, callback_group=self._sensor_cb_group)

            self._image_received = False
            self.enter = True
            self.start_get_roi = True

        threading.Thread(target=self._enter_go_home_and_refresh, daemon=True).start()
        response.success = True
        response.message = "start"
        return response

    def _enter_go_home_and_refresh(self):
        try:
            self.go_home()
        except Exception as e:
            self.get_logger().error(f'enter 回零线程异常: {e}')
        try:
            if self._refresh_robot_endpoint(force=True):
                # self.get_logger().info('回零完成，endpoint 已更新，深度定位可用')
                pass
            else:
                self.get_logger().warning('回零后仍无法获取 endpoint，将依赖 RGB 投影')
        except Exception as e:
            self.get_logger().error(f'enter 更新 endpoint 失败: {e}')

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "exit  object transport")
        with self.lock:
            self._init_parameters()
            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None
            if self.camera_info_sub is not None:
                self.destroy_subscription(self.camera_info_sub)
                self.camera_info_sub = None
            if self.depth_image_sub is not None:
                self.destroy_subscription(self.depth_image_sub)
                self.depth_image_sub = None
            if self.depth_cam_info_sub is not None:
                self.destroy_subscription(self.depth_cam_info_sub)
                self.depth_cam_info_sub = None
            self.depth_ready = False
            pick_and_place.interrupt()
        response.success = True
        response.message = "start"
        return response

    def enable_transport_srv_callback(self, request, response):
        with self.lock:
            if request.data:
                self.get_logger().info('\033[1;32m%s\033[0m' % 'enable  object transport')
                self.enable_transport = True
            else:
                self.get_logger().info('\033[1;32m%s\033[0m' % 'exit  object transport')
                pick_and_place.interrupt()
                self.enable_transport = False
        response.success = True
        response.message = "start"
        return response

    def abort_srv_callback(self, request, response):
        self.get_logger().info('[external_query] ~/abort')
        with self.lock:
            pick_and_place.interrupt()
            self.enable_transport = False
            self.start_transport = False
            self.action_list = []
            self.target = []
        msg = Bool()
        msg.data = True
        self.transport_finished_pub.publish(msg)
        response.success = True
        response.message = "aborted"
        return response

    def set_pick_position_srv_callback(self, request, response):
        with self.lock:
            # self.get_logger().info('\033[1;32mset pick: %s\033[0m' % request)
            label = (request.label or '').strip().lower()
            if label == 'restore_pick':
                self.action_list.append(['restore_pick', request.box])
            else:
                self.action_list.append(['pick', request.box])
        response.success = True
        response.message = "start"
        return response

    @staticmethod
    def _record_label_matches(query_label, stored_label):
        query_l = (query_label or '').lower().strip()
        stored_l = (stored_label or '').lower().strip()
        if not query_l or not stored_l:
            return False
        return query_l in stored_l or stored_l in query_l

    def _upsert_record_position(self, label, target):
        with self.lock:
            for i, entry in enumerate(self.record_position):
                if self._record_label_matches(label, entry[0]):
                    self.record_position[i] = [label, target]
                    return
            self.record_position.append([label, target])

    def clear_record_position_srv_callback(self, request, response):
        with self.lock:
            n = len(self.record_position)
            self.record_position = []
            self.action_list = [
                a for a in self.action_list if not (a and a[0] == 'record')
            ]
        self.get_logger().info(f'已清空 record_position（{n} 条）')
        return response

    def record_position_srv_callback(self, request, response):
        with self.lock:
            # self.get_logger().info('\033[1;32mrecord: %s\033[0m' % request)
            self.action_list.append(['record', request.label, request.box])
        response.success = True
        response.message = "start"
        return response

    def set_place_position_srv_callback(self, request, response):
        with self.lock:
            # self.get_logger().info('\033[1;32mset place: %s\033[0m' % request)
            if request.label and not request.box:
                self.action_list.append(['restore', request.label])
            else:
                self.action_list.append(['place', request.offset, request.box])
        response.success = True
        response.message = "start"
        return response

    def get_roi(self):
        transform_path = os.path.join(self._calibration_config_dir, 'transform.yaml')
        with open(transform_path, 'r') as f:
            config = yaml.safe_load(f)
        extristric_yaml = np.array(config['extristric'], dtype=np.float64)

        while self.intrinsic is None:
            self.get_logger().info('waiting for camera info')
            time.sleep(0.1)

        with self.lock:
            self.white_area_center = np.array(
                config['white_area_pose_world'], dtype=np.float64).reshape(4, 4)
            self.plane = tuple(config['plane'])
            self.corners = np.array(config['corners'])
            self.extristric_yaml = extristric_yaml
            self.extristric_arr = extristric_yaml.copy()
            tvec = extristric_yaml[:1]
            rmat = extristric_yaml[1:]
            self.extristric = (tvec, rmat)

    def get_object_pixel_position(self, image, roi):
        bbox_xyxy = np.asarray(roi, dtype=np.int32).ravel()
        everything_results = self.predictor(image)
        results = prompt(everything_results, bboxes=[bbox_xyxy], log=self.get_logger())
        
        # Get mask from results and ensure it matches original image size
        result = results[0]
        if len(result) == 0 or result.masks is None:
            self.get_logger().warning(f'FastSAM 未在框内生成掩码: {bbox_xyxy.tolist()}')
            return False
        mask = result.masks.data
        
        # Use YOLO's scale_masks function to properly scale mask to original image size
        if mask.shape[1:] != result.orig_shape:
            mask = scale_masks(mask[None], result.orig_shape)[0]

        mask = mask.cpu().numpy()
        if mask.ndim == 3 and mask.shape[0] == 1:  # Might be (1, H, W), need to remove the first dimension. 可能是 (1, H, W) 需要去掉第一维
            mask = mask[0]

        mask = (mask * 255).astype(np.uint8)

        # cv2.imshow('mask', mask)
        # cv2.waitKey(1)
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            grasp_point = (cx, cy)
            cv2.circle(image, grasp_point, 5, (255, 0, 0), -1)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            areaMaxContour, area_max = common.get_area_max_contour(contours)
            if areaMaxContour is not None:
                rect = cv2.minAreaRect(areaMaxContour)  # Obtain the minimum bounding rectangle 获取最小外接矩形
                # In version 4.5, w is defined as the edge that first aligns with the x-axis under clockwise rotation; angle is the clockwise rotation from the x-axis, ranging in (0, 90].
                rect_center, (width, height), angle = rect

                bbox = np.intp(cv2.boxPoints(rect))
                cv2.drawContours(image, [bbox], -1, (0, 255, 255), 2, cv2.LINE_AA)  # Draw rectangle contour. 绘制矩形轮廓
                
                _r = np.asarray(roi).ravel()
                cv2.rectangle(image, (int(_r[0]), int(_r[1])), (int(_r[2]), int(_r[3])), (255, 0, 0), 2, 1)
                # cv2.imshow('img', image)
                # cv2.waitKey(1)
                target_points_data = ["target", 0, rect_center, (width, height), angle]
                return grasp_point, target_points_data
        return False

    def _get_depth_frame(self):
        try:
            return self.depth_image_queue.get(block=True, timeout=0.5)
        except queue.Empty:
            return None, None

    def _prepare_depth_for_pose(self, depth_image, camera_info, depth_camera_info, bgr_image=None):
        """与标定 ColorPick 一致：工作区多边形 ROI + 统一平面高度。"""
        if self.corners is None or not hasattr(self, 'extristric_arr'):
            return depth_image, None
        if bgr_image is None:
            bgr_image = np.zeros(
                (depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
        return utils.create_roi_mask(
            depth_image.copy(),
            bgr_image,
            self.corners,
            camera_info,
            self.extristric_arr,
            400,
            utils.WORK_AREA_PLANE_HEIGHT,
            depth_camera_info,
        )

    @staticmethod
    def _rgb_pixel_to_depth_pixel(px, py, rgb_camera_info, depth_camera_info):
        pts = utils.remap_points_rgb_to_depth(
            np.array([[float(px), float(py)]]),
            rgb_camera_info,
            depth_camera_info,
        )
        return int(pts[0, 0]), int(pts[0, 1])

    @staticmethod
    def _median_depth_at(depth_image, x, y, roi=5):
        h, w = depth_image.shape[:2]
        x1, x2 = max(0, x - roi), min(w, x + roi + 1)
        y1, y2 = max(0, y - roi), min(h, y + roi + 1)
        patch = depth_image[y1:y2, x1:x2]
        valid = patch[np.logical_and(patch > 0, patch < 40000)]
        if len(valid) == 0:
            return 0
        return int(np.median(valid))

    @staticmethod
    def _apply_depth_angle(object_angle, depth_angle):
        if isinstance(object_angle, (list, tuple)) and len(object_angle) >= 1:
            fused = list(object_angle)
            fused[-1] = depth_angle
            return fused
        return depth_angle

    def _world_z_from_plane_offset(self, plane_offset_m):
        """plane_values 为相对拟合桌面的高度(m)，叠加 WORK_AREA_PLANE_HEIGHT 与 Z 补偿。"""
        plane = utils.WORK_AREA_PLANE_HEIGHT
        return plane + float(plane_offset_m) + self.DEPTH_Z_COMPENSATION

    @staticmethod
    def _plane_offset_at_pixel(plane_values, depth_x, depth_y, roi=5):
        h, w = plane_values.shape[:2]
        x1, x2 = max(0, depth_x - roi), min(w, depth_x + roi + 1)
        y1, y2 = max(0, depth_y - roi), min(h, depth_y + roi + 1)
        patch = plane_values[y1:y2, x1:x2]
        valid = np.isfinite(patch)
        if not np.any(valid):
            return None
        return float(np.median(patch[valid]))

    def get_world_position_from_depth(self, pixel_xy, contour_height=0.015):
        depth_image, depth_camera_info = self._get_depth_frame()
        if depth_image is None or depth_camera_info is None:
            self.get_logger().warning('深度图未就绪，无法计算三维位姿')
            return None
        if self.plane is None or self.endpoint is None:
            self.get_logger().warning('plane/endpoint 未初始化')
            return None
        with self.lock:
            rgb_info = self.latest_camera_info
        if rgb_info is None:
            self.get_logger().warning('RGB camera_info 未就绪')
            return None

        rgb_x, rgb_y = int(pixel_xy[0]), int(pixel_xy[1])
        depth_x, depth_y = self._rgb_pixel_to_depth_pixel(
            rgb_x, rgb_y, rgb_info, depth_camera_info)

        depth_image, _ = self._prepare_depth_for_pose(
            depth_image, rgb_info, depth_camera_info)

        depth_mm = self._median_depth_at(depth_image, depth_x, depth_y)
        if depth_mm <= 0:
            self.get_logger().warning(
                f'抓点深度无效 rgb=({rgb_x},{rgb_y}) depth=({depth_x},{depth_y})')
            return None

        plane_values = utils.get_plane_values(depth_image, self.plane, depth_camera_info.k)
        contours = utils.extract_contours(plane_values, contour_height)
        angle = 0.0
        best_cnt = None
        min_dist = float('inf')
        for cnt in contours:
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            dist = np.hypot(cx - depth_x, cy - depth_y)
            if dist < min_dist:
                min_dist = dist
                best_cnt = cnt
        if best_cnt is not None:
            _, _, angle = cv2.minAreaRect(best_cnt)

        position = utils.calculate_world_position(
            depth_x, depth_y, depth_mm, self.plane, self.endpoint,
            self.hand2cam_tf_matrix, depth_camera_info.k,
        )
        config_data = self._load_calibration_params()
        offset = tuple(config_data['depth']['offset'])
        scale = tuple(config_data['depth']['scale'])
        position[0] = position[0] * scale[0] + offset[0]
        position[1] = position[1] * scale[1] + offset[1]
        plane_offset = self._plane_offset_at_pixel(plane_values, depth_x, depth_y)
        if plane_offset is not None:
            position[2] = self._world_z_from_plane_offset(plane_offset)
        else:
            position[2] = position[2] * scale[2] + offset[2]
        self.get_logger().info(
            f'深度采样 rgb=({rgb_x},{rgb_y}) depth=({depth_x},{depth_y}) '
            f'mm={depth_mm} plane_off={plane_offset} xyz={position.tolist()}')
        return position, angle

    def _build_arm_target(self, world_position, grasp_point_pixel, object_angle, action_type):
        yaw = math.degrees(math.atan2(world_position[1], world_position[0]))
        if world_position[0] < 0:
            if world_position[1] < 0:
                yaw += 180
            else:
                yaw -= 180
        points = []
        gripper_size = self.gripper_pixel_size
        optimal_result = calculate_grasp_yaw.calculate_gripper_yaw_angle(
            object_angle, points, gripper_size, yaw)
        if optimal_result is None:
            yaw = yaw + object_angle[-1]
        else:
            yaw, _, _ = optimal_result
        yaw_servo_value = 500 + int(yaw / 240 * 1000)
        gripper_angle = 540 if action_type in ('pick', 'restore_pick') else 500
        return [world_position, grasp_point_pixel, yaw_servo_value, gripper_angle, action_type]

    @staticmethod
    def _normalize_xyxy_box(box):
        b = np.asarray(box, dtype=np.float64).ravel()
        if len(b) < 4:
            return None
        x1, y1, x2, y2 = b[:4]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return int(x1), int(y1), int(x2), int(y2)

    def _estimate_place_surface_z_world(self, place_box_xyxy, rgb_camera_info):
        """根据放置目标物体 VLLM 框内深度，估计其顶面世界坐标 Z。"""
        box = self._normalize_xyxy_box(place_box_xyxy)
        if box is None:
            return None
        if not self.get_parameter('use_depth').value or not self.depth_ready:
            return None
        if self.plane is None or self.endpoint is None or not self._refresh_robot_endpoint():
            return None
        depth_image, depth_camera_info = self._get_depth_frame()
        if depth_image is None or depth_camera_info is None:
            return None

        rgb_x1, rgb_y1, rgb_x2, rgb_y2 = box
        corners_rgb = np.array(
            [[rgb_x1, rgb_y1], [rgb_x2, rgb_y1], [rgb_x2, rgb_y2], [rgb_x1, rgb_y2]],
            dtype=np.float64,
        )
        corners_depth = utils.remap_points_rgb_to_depth(
            corners_rgb, rgb_camera_info, depth_camera_info)
        h_img, w_img = depth_image.shape[:2]
        x1 = int(np.clip(corners_depth[:, 0].min(), 0, w_img - 1))
        x2 = int(np.clip(corners_depth[:, 0].max(), 0, w_img - 1))
        y1 = int(np.clip(corners_depth[:, 1].min(), 0, h_img - 1))
        y2 = int(np.clip(corners_depth[:, 1].max(), 0, h_img - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        depth_image, _ = self._prepare_depth_for_pose(
            depth_image, rgb_camera_info, depth_camera_info)
        plane_values = utils.get_plane_values(depth_image, self.plane, depth_camera_info.k)
        roi_pv = plane_values[y1:y2, x1:x2]
        roi_depth = depth_image[y1:y2, x1:x2]
        valid = np.logical_and(roi_depth > 0, roi_depth < 40000)
        if not np.any(valid):
            return None

        # 俯视相机：深度更小 = 物理更高；取物体区域内偏小的深度分位作为顶面
        table_ref = float(np.percentile(roi_pv[valid], 15))
        top_ref = float(np.percentile(roi_pv[valid], 75))
        if top_ref <= table_ref + 0.003:
            obj_mask = valid
        else:
            obj_thresh = table_ref + (top_ref - table_ref) * 0.45
            obj_mask = valid & (roi_pv >= obj_thresh)
            if not np.any(obj_mask):
                obj_mask = valid

        top_plane_offset = float(np.percentile(roi_pv[obj_mask], 12))
        plane = utils.WORK_AREA_PLANE_HEIGHT
        z = self._world_z_from_plane_offset(top_plane_offset)
        z_lo = plane + 0.003
        z = float(max(z, z_lo))
        self.get_logger().info(
            f'放置目标顶面: z={z:.4f} 框={box} plane_off={top_plane_offset:.4f} '
            f'(plane {table_ref:.4f}~{top_ref:.4f}, clearance={self.PLACE_SURFACE_CLEARANCE}, '
            f'bias={self.PLACE_Z_BIAS}, z_comp={self.DEPTH_Z_COMPENSATION})')
        return z

    def _clamp_pick_place_z(self, world_position, action_type):
        if action_type == 'place':
            z = float(world_position[2])
            plane = utils.WORK_AREA_PLANE_HEIGHT
            z_lo = plane + 0.003
            world_position[2] = max(z_lo, z)
        return world_position

    def _resolve_world_pose(
            self, grasp_point_pixel, object_angle, bgr_image, camera_info, action_type,
            place_box_xyxy=None):
        place_target_z = None
        if action_type == 'place' and place_box_xyxy is not None:
            place_surface_z = self._estimate_place_surface_z_world(
                place_box_xyxy, camera_info)
            if place_surface_z is not None:
                place_target_z = (
                    place_surface_z
                    + self.PLACE_SURFACE_CLEARANCE
                    + self.PLACE_Z_BIAS
                )

        pick_height = utils.WORK_AREA_PLANE_HEIGHT
        rgb_height = pick_height if action_type in ('pick', 'restore_pick') else (
            place_target_z if place_target_z is not None else pick_height)
        distortion = None
        if camera_info is not None and len(camera_info.d) >= 4:
            distortion = np.array(camera_info.d)
        rgb_position, _ = self.get_object_world_position(
            grasp_point_pixel, self.intrinsic, self.extristric,
            self.white_area_center,
            height=rgb_height,
            distortion=distortion,
        )

        if (self.get_parameter('use_depth').value
                and self.depth_ready
                and self.endpoint is not None
                and self._refresh_robot_endpoint()):
            try:
                result = self.get_world_position_from_depth(
                    grasp_point_pixel, contour_height=0.015)
                if result is not None:
                    depth_position, depth_angle = result
                    if isinstance(depth_position, np.ndarray):
                        depth_position = depth_position.tolist()
                    if self._is_world_position_plausible(depth_position):
                        fused_angle = self._apply_depth_angle(object_angle, depth_angle)
                        if self.get_parameter('depth_fuse_rgb_xy').value:
                            world_position = [
                                float(rgb_position[0]),
                                float(rgb_position[1]),
                                float(depth_position[2]),
                            ]
                            self.get_logger().info(
                                f'深度+RGB融合 {action_type}: rgb_xy='
                                f'[{rgb_position[0]:.4f},{rgb_position[1]:.4f}] '
                                f'depth_xyz={depth_position} -> fused={world_position}')
                        else:
                            world_position = list(depth_position)
                            fused_angle = self._apply_depth_angle(
                                object_angle, depth_angle)
                            self.get_logger().info(
                                f'深度定位 {action_type}: xyz={world_position}')
                        if action_type == 'place' and place_target_z is not None:
                            world_position[2] = place_target_z
                        if action_type == 'restore_pick':
                            world_position[0] += self.RESTORE_PICK_X_BIAS
                        world_position = self._clamp_pick_place_z(
                            world_position, action_type)
                        if self._is_world_position_plausible(world_position):
                            return self._build_arm_target(
                                world_position, grasp_point_pixel, fused_angle, action_type)
                        self.get_logger().warning(
                            f'融合后世界坐标异常 xyz={world_position}，使用 RGB 投影')
                    else:
                        self.get_logger().warning(
                            f'深度世界坐标异常 xyz={depth_position}，使用 RGB 投影')
            except Exception as e:
                self.get_logger().warning(f'深度定位失败，使用 RGB 投影: {e}')

        world_position = list(rgb_position)
        if action_type in ('pick', 'restore_pick'):
            world_position[2] = self._world_z_from_plane_offset(0.0)
        if action_type == 'restore_pick':
            world_position[0] += self.RESTORE_PICK_X_BIAS
        world_position = self._clamp_pick_place_z(world_position, action_type)
        self.get_logger().info(f'RGB 投影 {action_type}: xyz={world_position}')
        return self._build_arm_target(
            world_position, grasp_point_pixel, object_angle, action_type)

    def _compose_depth_display(self, bgr_image, camera_info, grasp_pixel=None):
        if not self.depth_ready:
            panel = np.zeros((bgr_image.shape[0], bgr_image.shape[1] // 2, 3), dtype=np.uint8)
            cv2.putText(panel, 'depth: waiting', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return panel
        depth_image, depth_camera_info = self._get_depth_frame()
        if depth_image is None or depth_camera_info is None:
            return None
        if self.corners is None or not hasattr(self, 'extristric_arr'):
            return None
        max_dist = 400
        depth_copy, depth_imgpts = self._prepare_depth_for_pose(
            depth_image, camera_info, depth_camera_info, bgr_image)
        sim = (1 - np.clip(depth_copy, 0, max_dist).astype(np.float64) / max_dist) * 255
        depth_color = cv2.applyColorMap(sim.astype(np.uint8), cv2.COLORMAP_JET)
        if depth_imgpts is not None:
            cv2.drawContours(depth_color, [depth_imgpts], -1, (0, 255, 255), 2, cv2.LINE_AA)
        if grasp_pixel is not None:
            gp = np.array([[float(grasp_pixel[0]), float(grasp_pixel[1])]])
            if (depth_camera_info.width != camera_info.width
                    or depth_camera_info.height != camera_info.height):
                gp = utils.remap_points_rgb_to_depth(gp, camera_info, depth_camera_info)
            gx, gy = int(gp[0, 0]), int(gp[0, 1])
            if 0 <= gx < depth_color.shape[1] and 0 <= gy < depth_color.shape[0]:
                cv2.circle(depth_color, (gx, gy), 8, (0, 0, 255), -1)
        return depth_color

    @classmethod
    def _concat_rgb_depth_panels(cls, rgb_bgr, depth_bgr):
        crop_top, crop_end = cls.DISPLAY_CROP_TOP, cls.DISPLAY_CROP_END
        if depth_bgr.shape[0] != rgb_bgr.shape[0]:
            depth_bgr = cv2.resize(
                depth_bgr,
                (depth_bgr.shape[1], rgb_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        rgb_crop = rgb_bgr[crop_top:crop_end, :]
        dcm_crop = depth_bgr[crop_top:crop_end, :]
        if dcm_crop.shape[0] != rgb_crop.shape[0]:
            dcm_crop = cv2.resize(
                dcm_crop,
                (dcm_crop.shape[1], rgb_crop.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return np.concatenate([rgb_crop, dcm_crop], axis=1)

    def _undistort_grasp_pixel(self, px, py, intrinsic, distortion):
        if distortion is None or len(np.asarray(distortion).ravel()) < 4:
            return float(px), float(py)
        K = np.array(intrinsic, dtype=np.float64).reshape(3, 3)
        D = np.asarray(distortion, dtype=np.float64).ravel()
        pt = cv2.undistortPoints(
            np.array([[[float(px), float(py)]]], dtype=np.float64), K, D, P=K)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    def get_object_world_position(
            self, position, intrinsic, extristric, white_area_center,
            height=0.025, distortion=None):
        del extristric
        ext = getattr(self, 'extristric_yaml', self.extristric_arr)
        tvec = ext[:1]
        rmat = ext[1:]
        tvec, rmat = common.extristric_plane_shift(
            np.array(tvec).reshape((3, 1)), np.array(rmat), height)
        projection_matrix = np.row_stack(
            (np.column_stack((rmat, tvec)), np.array([[0, 0, 0, 1]])))

        px, py = self._undistort_grasp_pixel(
            position[0], position[1], intrinsic, distortion)
        world_pose = common.pixels_to_world([[px, py]], intrinsic, projection_matrix)[0]
        temp = world_pose[0]
        world_pose[0] = world_pose[1]
        world_pose[1] = -temp
        position = white_area_center[:3, 3] + world_pose
        position[2] = height

        config_data = self._load_calibration_params()
        offset = tuple(config_data['pixel']['offset'])
        scale = tuple(config_data['pixel']['scale'])
        for i in range(3):
            position[i] = position[i] * scale[i]
            position[i] = position[i] + offset[i]
        return position, projection_matrix

    def main(self):
        while True:
            if self.enter:
                try:
                    bgr_image = self.image_queue.get(block=True, timeout=1)
                except queue.Empty:
                    self.get_logger().warning(
                        f'未收到图像，请检查 {self.get_parameter("rgb_image_topic").value}'
                    )
                    continue
                with self.lock:
                    if self.latest_camera_info is None:
                        continue
                    camera_info = self.latest_camera_info
                    self.intrinsic = np.matrix(camera_info.k).reshape(1, -1, 3)
                    if self.start_get_roi:
                        self.get_roi()
                        self.start_get_roi = False

                if self.enable_transport:
                    with self.lock:
                        if self.action_list:
                            box_info = self.action_list[0]
                            if box_info[0] in ('pick', 'restore_pick'):
                                self._draw_vllm_box(bgr_image, box_info)
                                pick_kind = box_info[0]
                                if not self.target:
                                    try:
                                        grasp_point_pixel, object_angle = self.get_object_pixel_position(
                                            bgr_image, box_info[1])
                                        if grasp_point_pixel:
                                            self.target = self._resolve_world_pose(
                                                grasp_point_pixel, object_angle, bgr_image,
                                                camera_info, pick_kind)
                                            box = np.asarray(box_info[1]).ravel()
                                            x_bias = (
                                                self.RESTORE_PICK_X_BIAS
                                                if pick_kind == 'restore_pick' else 0.0
                                            )
                                            self.get_logger().info(
                                                f'{pick_kind} 定位: vllm框={box.tolist()} '
                                                f'抓点像素={grasp_point_pixel} '
                                                f'世界坐标={self.target[0]} '
                                                f'x_bias={x_bias}')
                                            self.start_stamp = time.time()
                                    except Exception as e:
                                        self.get_logger().error(f'{pick_kind} 定位失败: {e}')

                                else:
                                    if time.time() - self.start_stamp > 2:
                                        self.start_transport = True        
                                        self.enable_transport = False
                                    cv2.circle(bgr_image, (int(self.target[1][0]), int(self.target[1][1])), 10, (255, 0, 0), -1)
                            elif box_info[0] == 'place':
                                self._draw_vllm_box(bgr_image, box_info)
                                if not self.target:
                                    try:
                                        grasp_point_pixel, object_angle = self.get_object_pixel_position(
                                            bgr_image, box_info[-1])
                                        if grasp_point_pixel:
                                            self.target = self._resolve_world_pose(
                                                grasp_point_pixel, object_angle, bgr_image,
                                                camera_info, 'place',
                                                place_box_xyxy=box_info[-1])
                                            self.target[0][0] += box_info[-2][0]
                                            self.target[0][1] += box_info[-2][1]
                                            self.get_logger().info(
                                                f'place target: xyz={self.target[0]} '
                                                f'offset_xy={box_info[-2]}')
                                            self.start_stamp = time.time()
                                    except Exception as e:
                                        self.get_logger().error(f'place 定位失败: {e}')
                                else:
                                    if time.time() - self.start_stamp > 2:
                                        self.start_transport = True
                                        self.enable_transport = False
                                    cv2.circle(bgr_image, (int(self.target[1][0]), int(self.target[1][1])), 10, (255, 0, 0), -1)
                            elif box_info[0] == 'record':
                                # remember 记录初位：不画 VLLM 绿框，避免与框选标识混淆
                                try:
                                    grasp_point_pixel, object_angle = self.get_object_pixel_position(
                                        bgr_image, box_info[-1])
                                    if grasp_point_pixel:
                                        self.target = self._resolve_world_pose(
                                            grasp_point_pixel, object_angle, bgr_image,
                                            camera_info, 'place',
                                            place_box_xyxy=box_info[-1])
                                        self._upsert_record_position(box_info[1], self.target)
                                        del self.action_list[0]
                                        if not self.action_list:
                                            self.get_logger().info('记录位置完成')
                                            self.enable_transport = False
                                            self.target = []
                                except Exception as e:
                                    self.get_logger().error(f'record 定位失败: {e}')
                            elif box_info[0] == 'restore':
                                matched = None
                                for i in range(len(self.record_position) - 1, -1, -1):
                                    entry = self.record_position[i]
                                    if self._record_label_matches(box_info[1], entry[0]):
                                        matched = entry
                                        break
                                if matched is not None:
                                    self.target = copy.deepcopy(matched[1])
                                    self.target[-1] = 'restore'
                                    self.start_transport = True
                                    self.enable_transport = False
                                else:
                                    self.get_logger().error(
                                        f'restore 未匹配 record_position: 查询={box_info[1]!r}, '
                                        f'已记录={[x[0] for x in self.record_position]}')
                                    del self.action_list[0]
                                    msg = Bool()
                                    msg.data = True
                                    self.transport_finished_pub.publish(msg)
                        else:
                            msg = Bool()
                            msg.data = True
                            self.transport_finished_pub.publish(msg)
                if self.track_window:  # 绘制最终确定的选框 (红色)
                    cv2.rectangle(bgr_image, (self.track_window[0], self.track_window[1]),
                                  (self.track_window[2], self.track_window[3]), (0, 0, 255), 2)
                elif self.selection:  # 绘制正在拖动的选框 (黄色)
                    cv2.rectangle(bgr_image, (self.selection[0], self.selection[1]), (self.selection[2], self.selection[3]),
                                  (0, 255, 255), 2)

                self.fps.update()
                display_image = self.fps.show_fps(bgr_image)
                if self.get_parameter('display').value:
                    if self.get_parameter('use_depth').value:
                        grasp_px = None
                        if self.target and len(self.target) > 1:
                            grasp_px = self.target[1]
                        depth_panel = self._compose_depth_display(
                            bgr_image, camera_info, grasp_px)
                        if depth_panel is not None:
                            display_image = self._concat_rgb_depth_panels(
                                display_image, depth_panel)
                    cv2.imshow('image', display_image)
                    cv2.waitKey(1)
                    if not self.set_above:
                        cv2.moveWindow('image', 1920 - 640 * 2, 0)
                        try:
                            if shutil.which('wmctrl'):
                                os.system('wmctrl -r image -b add,above')
                        except Exception:
                            pass
                        self.set_above = True
            else:
                time.sleep(0.1)
        cv2.destroyAllWindows()

    def transport_thread(self):
        while True:
            if self.start_transport:
                config_data = self._load_calibration_params()
                offset = tuple(config_data['kinematics']['offset'])
                scale = tuple(config_data['kinematics']['scale'])
                p = copy.deepcopy(self.target)
                for i in range(3):
                    p[0][i] = p[0][i] * scale[i]
                    p[0][i] = p[0][i] + offset[i]
                self.get_logger().info(
                    f'开始搬运 {self.target[-1]}: xyz={p[0]}, yaw={p[2]}')
                finish = False
                try:
                    if self.target[-1] in ('pick', 'restore_pick'):
                        finish = pick_and_place.pick_without_back(
                            p[0], 80, p[2], 500, self.PICK_GRIPPER_DEPTH,
                            self.joints_pub, self.kinematics_client,
                            offset_z=self.PICK_APPROACH_OFFSET_Z)
                    elif self.target[-1] in ('place', 'restore'):
                        if self.target[-1] == 'restore':
                            p[0][2] -= 0.015
                        finish = pick_and_place.place(
                            p[0], 80, p[2], 400,
                            self.joints_pub, self.kinematics_client,
                            offset_z=0.015)
                except Exception as e:
                    self.get_logger().error(f'搬运执行异常: {e}')
                if finish:
                    self.go_home(False)
                else:
                    self.get_logger().error(
                        f'搬运失败(IK 无解或舵机未反馈): {self.target[-1]}')
                    self.go_home()
                try:
                    self._refresh_robot_endpoint(force=True)
                except Exception as e:
                    self.get_logger().warning(f'搬运后更新 endpoint 失败: {e}')
                with self.lock:
                    self.target = []
                    if self.action_list:
                        del self.action_list[0]
                self.enable_transport = True
                self.start_transport = False
                msg = Bool()
                msg.data = True
                self.transport_finished_pub.publish(msg)
            else:
                time.sleep(0.1)

    def camera_info_callback(self, msg):
        with self.lock:
            self.latest_camera_info = msg

    def depth_camera_info_callback(self, msg):
        with self.depth_lock:
            self.latest_depth_cam_info = msg

    def depth_image_callback(self, ros_depth_image):
        with self.depth_lock:
            cam_info = self.latest_depth_cam_info
        if cam_info is None:
            return
        depth_image = np.ndarray(
            shape=(ros_depth_image.height, ros_depth_image.width),
            dtype=np.uint16, buffer=ros_depth_image.data)
        if self.depth_image_queue.full():
            self.depth_image_queue.get()
        self.depth_image_queue.put((depth_image, cam_info))
        # if not self.depth_ready:
        #     self.get_logger().info('已收到深度图，深度定位/预览已启用')
        self.depth_ready = True

    def image_callback(self, ros_rgb_image):
        try:
            encoding = ros_rgb_image.encoding or 'bgr8'
            if encoding == 'rgb8':
                cv_image = self.bridge.imgmsg_to_cv2(ros_rgb_image, 'rgb8')
                bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(ros_rgb_image, 'bgr8')
                bgr_image = np.array(cv_image, dtype=np.uint8)
        except Exception as e:
            self.get_logger().error(f'图像解码失败: {e}')
            return
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(bgr_image)
        if not self._image_received:
            self._image_received = True
            # self.get_logger().info(
            #     f'已收到相机图像 {ros_rgb_image.width}x{ros_rgb_image.height} '
            #     f'({self.get_parameter("rgb_image_topic").value})')

def main():
    node = ObjectTransport('object_transport')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()
