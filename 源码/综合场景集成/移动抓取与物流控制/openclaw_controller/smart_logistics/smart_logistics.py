#!/usr/bin/env python3
# encoding: utf-8
import math, os, threading, time

import rclpy
import sdk.common as common
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from interfaces.srv import SetString
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sdk.pid import PID
from sensor_msgs.msg import Image
from servo_controller.action_group_controller import ActionGroupController
from servo_controller_msgs.msg import ServosPosition
from std_srvs.srv import Trigger

from openclaw_controller.smart_logistics.smart_logistics_vision import SmartLogisticsVisionMixin


class SmartLogistics(SmartLogisticsVisionMixin, Node):
    default_config_path = '/home/ubuntu/ros2_ws/src/openclaw_controller/openclaw_controller/smart_logistics/config/smart_logistics.yaml'

    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.cb_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        self.action_lock = threading.Lock()

        self.target_color = 'red'
        self.latest_image = None
        self.display_result = None
        self.arm_state = 'init'
        self.running = True
        self.color_names = ['red', 'green', 'blue']

        self.camera_type = 'Stereo' if os.environ.get('DEPTH_CAMERA_TYPE') in ['ascamera', 'aurora'] else 'Mono'
        self.machine_type = os.environ.get('MACHINE_TYPE', '')
        self.lab_data = common.get_yaml_data('/home/ubuntu/software/lab_tool/lab_config.yaml')

        self.load_params()

        self.linear_pid = PID(P=self.linear_pid_p, I=0, D=0)
        self.angular_pid = PID(P=self.angular_pid_p, I=0, D=0)
        self.yaw_pid = PID(P=self.yaw_pid_p, I=0, D=0)

        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.image_pub = self.create_publisher(Image, '~/image_result', 1)
        self.controller = ActionGroupController(self.joints_pub, self.action_group_path)

        self.create_subscription(Image, self.camera_topic, self.image_callback, 1, callback_group=self.cb_group)
        self.create_service(SetString, '~/set_color', self.on_set_color, callback_group=self.cb_group)
        self.create_service(Trigger, '~/calibrate_pick', self.on_calibrate_pick, callback_group=self.cb_group)
        self.create_service(Trigger, '~/calibrate_place', self.on_calibrate_place, callback_group=self.cb_group)
        self.create_service(Trigger, '~/pick', self.on_pick, callback_group=self.cb_group)
        self.create_service(Trigger, '~/place', self.on_place, callback_group=self.cb_group)
        self.create_service(Trigger, '~/get_arm_state', self.on_get_arm_state, callback_group=self.cb_group)
        self.create_service(Trigger, '~/count_color', self.on_count_color, callback_group=self.cb_group)
        self.create_service(Trigger, '~/init_pose', self.on_init_pose, callback_group=self.cb_group)
        self.create_service(Trigger, '~/init_finish', self.on_init_finish, callback_group=self.cb_group)

        self.controller_client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.controller_client.wait_for_service()
        self.controller.run_action(self.init_action)
        self.arm_state = 'init'

        if self.enable_display:
            threading.Thread(target=self.display_loop, daemon=True).start()

        self.get_logger().info('start')

    def load_params(self):
        self.camera_topic = self.get_param('camera_topic', '/depth_cam/rgb0/image_raw')
        self.enable_display = self.get_param('enable_display', True)
        self.roi = self.get_param('roi', [0.0, 0.0, 1.0, 1.0])
        self.pick_roi_scale = 0.8
        self.place_roi_scale = 0.5
        self.min_area = self.get_param('min_area', 300)
        self.config_path = self.get_param('config_path', self.default_config_path)
        config_params = common.get_yaml_data(self.config_path)['/**']['ros__parameters']

        self.pick_stop = self.get_param('pick_stop_pixel_coordinate', config_params['pick_stop_pixel_coordinate'])
        self.place_stop = self.get_param('place_stop_pixel_coordinate', config_params['place_stop_pixel_coordinate'])

        self.x_tolerance = 10
        self.y_tolerance = 10
        self.angle_tolerance = 3.0
        self.stable_count_target = 8
        self.approach_timeout = 25.0
        self.control_period = 0.05

        self.linear_pid_p = 0.0018
        self.angular_pid_p = 0.0023
        self.yaw_pid_p = 0.015
        self.max_linear = 0.15
        self.max_angular = 1.2
        self.max_yaw = 0.8

        self.action_group_path = self.get_param('action_group_path', '/home/ubuntu/software/arm_pc/ActionGroups')
        self.init_action = self.get_param('init_action', 'smart_logistics_pick_init')
        self.pick_action = self.get_param('pick_action', 'smart_logistics_pick')
        self.place_action = self.get_param('place_action', 'smart_logistics_place')
        self.pick_debug_action = self.get_param('pick_debug_action', 'smart_logistics_pick_debug')
        self.place_debug_action = self.get_param('place_debug_action', 'smart_logistics_place_debug')

    def get_param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    # ---- helpers ----
    def get_latest_image(self):
        with self.image_lock:
            if self.latest_image is None:
                return None
            return self.latest_image.copy()

    def stop_base(self):
        self.mecanum_pub.publish(Twist())

    def run_init_action(self):
        self.controller.run_action(self.init_action)
        self.arm_state = 'init'

    def action_exists(self, action_name):
        return os.path.exists(os.path.join(self.action_group_path, action_name + '.d6a'))

    def save_config_value(self, name, value):
        data = common.get_yaml_data(self.config_path)
        data['/**']['ros__parameters'][name] = value
        common.save_yaml_data(data, self.config_path)

    def detect_and_save(self, color, param_name, debug_action):
        """run debug posture -> user places block -> init posture -> detect & save stop pixel
        (执行debug姿势让出空间 -> 用户放置色块 -> init姿势夹爪归位 -> 检测并保存停靠坐标)"""
        # step 1: debug posture — arm clears space for user to place block (debug姿势让出空间)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'calibrate %s: running %s' % (param_name, debug_action))
        self.controller.run_action(debug_action)  # blocking (阻塞)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'calibrate %s: place %s block now...' % (param_name, color))
        time.sleep(5)  # wait for user to place block (等待用户放置色块)

        # step 2: init posture — arm moves to gripping position over the block (init姿势夹爪归位)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'calibrate %s: running %s' % (param_name, self.init_action))
        self.run_init_action()  # blocking (阻塞)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'calibrate %s: done, detecting...' % param_name)

        detection_count = 0
        start_time = time.time()
        pixel = None
        while rclpy.ok() and time.time() - start_time < 15.0:
            img = self.get_latest_image()
            if img is None:
                time.sleep(0.1)
                continue
            result = self.detect_target_color(img, color, self.get_task_roi(param_name))
            self.display_result = result
            if result['color'] != 'none':
                detection_count += 1
                if detection_count >= 10:
                    pixel = [int(result['center'][0]), int(result['center'][1])]
                    break
            else:
                detection_count = 0
            time.sleep(0.1)

        if pixel is None:
            self.get_logger().error('calibrate %s: failed to detect %s' % (param_name, color))
            self.run_init_action()
            self.display_result = None
            return False

        self.save_config_value(param_name, pixel)
        if param_name == 'pick_stop_pixel_coordinate':
            self.pick_stop = pixel
        else:
            self.place_stop = pixel
        self.get_logger().info('\033[1;32m%s\033[0m' % 'calibrate %s: saved %s: %s' % (param_name, param_name, pixel))

        self.stop_base()
        self.run_init_action()
        time.sleep(2)
        self.display_result = None
        return True

    # ---- service callbacks ----
    def on_init_finish(self, request, response):
        response.success = True
        return response

    def on_get_arm_state(self, request, response):
        response.success = True
        response.message = self.arm_state
        return response

    def on_count_color(self, request, response):
        image = self.get_latest_image()
        if image is None:
            response.success = False
            response.message = 'no image'
            return response
        counts, result = self.count_all_colors(image)
        self.display_result = result
        threading.Thread(target=self.clear_display_later, args=(result,), daemon=True).start()
        response.success = True
        response.message = ','.join(['%s:%d' % (color, counts[color]) for color in self.color_names])
        self.get_logger().info('count %s' % response.message)
        return response

    def on_set_color(self, request, response):
        color = request.data.strip().lower()
        if color not in self.color_names:
            response.success = False
            response.message = 'invalid color'
            return response
        self.target_color = color
        response.success = True
        response.message = color
        self.get_logger().info('set color: %s' % color)
        return response

    def on_calibrate_pick(self, request, response):
        """calibrate pick stop pixel only, no chassis movement, no action
        (仅校准pick停靠坐标，不移动底盘，不执行动作)"""
        if not self.action_lock.acquire(blocking=False):
            response.success = False
            response.message = 'busy'
            return response
        ok = self.detect_and_save('red', 'pick_stop_pixel_coordinate', self.pick_debug_action)
        self.action_lock.release()
        response.success = ok
        response.message = 'finish' if ok else 'no target'
        return response

    def on_calibrate_place(self, request, response):
        """calibrate place stop pixel only, no chassis movement, no action
        (仅校准place停靠坐标，不移动底盘，不执行动作)"""
        if not self.action_lock.acquire(blocking=False):
            response.success = False
            response.message = 'busy'
            return response
        ok = self.detect_and_save('red', 'place_stop_pixel_coordinate', self.place_debug_action)
        self.action_lock.release()
        response.success = ok
        response.message = 'finish' if ok else 'no target'
        return response

    def on_pick(self, request, response):
        """approach target using saved pick_stop then execute pick action
        (使用已保存的pick_stop对齐目标，然后执行夹取动作)"""
        if not self.action_lock.acquire(blocking=False):
            response.success = False
            response.message = 'busy'
            return response
        self.display_result = None
        self.get_logger().info('\033[1;32m%s\033[0m' % 'pick: approaching %s' % self.target_color)
        if not self.approach_target(self.target_color, self.pick_stop, align_angle=True):
            self.display_result = None
            self.action_lock.release()
            response.success = False
            response.message = 'approach failed'
            return response
        self.stop_base()
        time.sleep(0.5)
        self.display_result = None
        self.get_logger().info('\033[1;32m%s\033[0m' % 'pick: executing %s' % self.pick_action)
        self.arm_state = 'picking'
        self.controller.run_action(self.pick_action)
        self.arm_state = 'pick_finish'
        self.stop_base()
        self.display_result = None
        self.action_lock.release()
        response.success = True
        response.message = 'finish'
        return response

    def on_place(self, request, response):
        """approach target using saved place_stop then execute place action
        (使用已保存的place_stop对齐目标，然后执行放置动作)"""
        if not self.action_lock.acquire(blocking=False):
            response.success = False
            response.message = 'busy'
            return response
        self.display_result = None
        self.get_logger().info('\033[1;32m%s\033[0m' % 'place: approaching %s' % self.target_color)
        if not self.approach_target(self.target_color, self.place_stop, align_angle=False):
            self.display_result = None
            self.action_lock.release()
            response.success = False
            response.message = 'approach failed'
            return response
        self.stop_base()
        time.sleep(0.5)
        self.display_result = None
        self.get_logger().info('\033[1;32m%s\033[0m' % 'place: executing %s' % self.place_action)
        self.arm_state = 'placing'
        self.controller.run_action(self.place_action)
        self.arm_state = 'place_finish'
        self.run_init_action()
        self.stop_base()
        self.display_result = None
        self.action_lock.release()
        response.success = True
        response.message = 'finish'
        return response

    def on_init_pose(self, request, response):
        if not self.action_lock.acquire(blocking=False):
            response.success = False
            response.message = 'busy'
            return response
        self.stop_base()
        self.run_init_action()
        self.action_lock.release()
        response.success = True
        response.message = 'finish'
        return response

    # ---- chassis approach ----
    def approach_target(self, color, stop_pixel, align_angle=False):
        self.linear_pid.clear()
        self.angular_pid.clear()
        self.yaw_pid.clear()
        stable_count = 0
        start_time = time.time()

        while rclpy.ok() and time.time() - start_time < self.approach_timeout:
            image = self.get_latest_image()
            if image is None:
                self.stop_base()
                time.sleep(self.control_period)
                continue

            task_name = 'pick' if align_angle else 'place'
            result = self.detect_target_color(image, color, self.get_task_roi(task_name))
            self.display_result = result
            if result['color'] == 'none':
                stable_count = 0
                self.stop_base()
                time.sleep(self.control_period)
                continue

            twist = self.get_approach_twist(result, stop_pixel, align_angle)
            self.mecanum_pub.publish(twist)
            if self.is_twist_stopped(twist):
                stable_count += 1
                if stable_count >= self.stable_count_target:
                    self.stop_base()
                    return True
            else:
                stable_count = 0
            time.sleep(self.control_period)

        self.stop_base()
        return False

    def get_approach_twist(self, result, stop_pixel, align_angle):
        twist = Twist()
        center_x, center_y = result['center']
        stop_x, stop_y = stop_pixel
        x_error = center_x - stop_x
        y_error = center_y - stop_y

        if abs(y_error) > self.y_tolerance:
            self.linear_pid.SetPoint = stop_y
            self.linear_pid.update(center_y)
            twist.linear.x = float(max(-self.max_linear, min(self.max_linear, self.linear_pid.output)))

        if abs(x_error) > self.x_tolerance:
            self.angular_pid.SetPoint = stop_x
            self.angular_pid.update(center_x)
            twist.angular.z = float(max(-self.max_angular, min(self.max_angular, self.angular_pid.output)))

        if self.is_twist_stopped(twist) and align_angle and 'Mecanum' in self.machine_type:
            object_angle = result['angle']
            if object_angle < 40:
                object_angle += 90
            if abs(object_angle - 90) > self.angle_tolerance:
                self.yaw_pid.SetPoint = 90
                self.yaw_pid.update(object_angle)
                yaw = max(-self.max_yaw, min(self.max_yaw, self.yaw_pid.output))
                twist.linear.y = float(-0.6 * math.sin(yaw / 2))
                twist.angular.z = float(yaw)

        return twist

    def is_twist_stopped(self, twist):
        return abs(twist.linear.x) < 0.001 and abs(twist.linear.y) < 0.001 and abs(twist.angular.z) < 0.001


def main():
    node = SmartLogistics('smart_logistics')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.running = False
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
