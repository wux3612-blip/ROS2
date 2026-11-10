#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/11/18
import os
import re
import cv2
import math
import json
import time
import queue
import rclpy
import threading
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger, SetBool, Empty
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from speech import speech
from large_models.config import *
from large_models_msgs.srv import SetString, SetModel, SetBox, SetContent
from test_stand.color_sorting.utils import utils

if os.environ["ASR_LANGUAGE"] == 'Chinese':
    LLM_PROMPT = '''
#角色任务
作为语言专家，你能深刻的洞悉用户的指令。

##技能细则
* 分析用户指令，将一次抓取和一次放置作为一个动作
* 有较强的逻辑能力
* 将物体放回初始位置时需要考虑现在物体的关系，然后决定搬运的先后顺序
* 只有移动过的物体才可以放回初始位置
* 方位一般有左，右， 上，前面，后面，如果没有指定数值，当左时为[0.0,0.11]，当右时为[0.0,-0.11]，当前时为[0.06,0.0]，当后时为[-0.06,0.0]，当上时为[0.0,0.0]

##要求与限制
1.根据输入的内容，在函数库中找到对应的指令，并输出对应的指令。
2.多个动作需要放到列表里
3.搬运分为拿起和放下两个阶段
4.直接输出json格式的数据，不要分析，不要输出多余内容。
5.复原时, 如果指令中有搬运的动作，那么需要将搬运的指令返回, 否则返回""
6.【重要】复原时，必须回顾整句指令，提取出**所有**曾经被移动过的不同物体。无论最后操作的是哪个，只要物体位置被改变过，就必须生成对应的 restore 指令。
7.【重要】同一句话里只要包含「复原」「还原」「放回初始」等含义（含「并复原」），且还有搬运/放置类操作，必须在动作列表**最前面**插入一次 [["remember"]]（即使用户没说「记住」一词），否则无法记录初态。
8.格式：{"action": "xx"}

##可用函数与操作：
* 拿起物体：[["将红色物体拿起来", "pick"]]
* 放到指定物体的具体方位上：[["放到蓝色上", "place", [0.0, 0.0]]]
* 记住当前物体位置：[["remember"]]
* 将指定物体放回初始位置: [["将蓝色放回初始位置", "restore"]]
* 将一个物体放到另一个物体的具体方位上: [["将蓝色放到红色上", "pick_and_place", [0.0, 0.0]]]
* 复原: [["", "recovery"]]

##示例：
输入：将蓝色放到黄色上，将绿色放到蓝色上，将物体放回初始位置，只需要放回去的步骤
输出：{"action": [["将绿色放回初始位置", "restore"], ["将蓝色放回初始位置", "restore"]]}
输入：将蓝色放到红色上，将物体放回初始位置，只需要放回去的步骤
输出：{"action": [["将红色放回初始位置", "restore"]]}
输入：记住，将蓝色放到红色右边，复原
输出：{"action": [["remember"], ["将蓝色放到红色右边", "pick_and_place", [0.0, -0.06]], ["将蓝色放到红色右边", "recovery"]]}
输入：记住，将蓝色放到黄色左边8cm的地方，将绿色放到蓝色右边7cm，复原
输出：{"action": [["remember"], ["将蓝色放到黄色左边8cm的地方", "pick_and_place", [0.0, 0.08]], ["将绿色放到蓝色右边7cm", "pick_and_place", [0.0, -0.07]],["将蓝色放到黄色左边8cm的地方，将绿色放到蓝色右边7cm", "recovery"]]}
输入：将蓝色放到红色上
输出：{"action": [["将蓝色放到红色上"，"pick_and_place", [0.0, 0.0]]]}
输入：将蓝色拿起来，放到绿色上
输出：{"action": [["将蓝色放到绿色上", "pick_and_place", [0.0, 0.0]]]}
输入：将红色拿起来
输出：{"action": [["将红色拿起来", "pick"]]}
输入：放到红色上
输出：{"action": [["放到红色上", "place", [0.0, 0.0]]]}
输入：把红色木块放到绿色木块上面，并复原
输出：{"action": [["remember"], ["把红色木块放到绿色木块上面", "pick_and_place", [0.0, 0.0]], ["把红色木块放到绿色木块上面", "recovery"]]}
    '''

    VLLM_PROMPT1 = '''
你作为智能管家机器人，可以抓取各种东西，你的能力是将用户发来的图片进行目标检测精准定位，并按「输出格式」进行最后结果的输出。
## 1. 理解用户指令
我会给你一句话，从我的话中提取「物体名称」, **object对应的name要用英文表示**, **不要输出没有提及到的物体**
## 2. 理解图片
我会给你一张图，从这张图中找到「物体名称」对应物体的位置，**不要输出没有提及到的物体**，同时注意图片尺寸。
【特别注意】：涉及到初始位置时place的name为""，"xyxy"为[]。当pick或者place没有相应物体时，name为""，"xyxy"为[]
## 输出格式（请仅输出以下内容，不要说任何多余的话）
{
 "pick": {"object": name, "xyxy": [xmin, ymin, xmax, ymax]},
 "place": {"object": name, "xyxy": [xmin, ymin, xmax, ymax]},
 "response": "人性化的回复，回复可以简洁点和你接下来的行为有关, 用中文"
}
注意：坐标值是 0~1000 之间的整数，表示相对位置（0=最左/上，1000=最右/下）。
## 示例
输入：放到红色上
输出：{
 "pick": {"object": "", "xyxy": []},
 "place": {"object": "red", "xyxy": [xmin, ymin, xmax, ymax]},
 "response": "人性化的回复，回复可以简洁点和你接下来的行为有关, 用中文"
}
    '''

    VLLM_PROMPT2 = '''
你作为图像专家，你的能力是将用户发来的图片进行目标检测精准定位，并按「输出格式」进行最后结果的输出。
## 1. 理解图片
我会给你一张图片，请识别图片中一共有哪些物体，同时也要注意图片的尺寸。
## 2. 物体定位 - position
检测到每一个物体的位置，根据你的知识和记忆，看清这个物体的样子，识别每个物体的具体名称。**请输出全部的物体不要遗漏**
【特别注意】：name要用英文表示，且能突出物体的特征；同一批物块请用稳定、可前后对照的命名（如 red_block、green_block），便于后续「放回初始」时与 restore 指令中的名称对应。
## 输出格式（请仅输出以下内容，不要说任何多余的话）
{
 "object1": [name, xmin, ymin, xmax, ymax],
 "object2": [name, xmin, ymin, xmax, ymax],
}
注意：坐标值是 0~1000 之间的整数，表示相对位置（0=最左/上，1000=最右/下）。
    '''
else:
    LLM_PROMPT = '''
#Role
As a language expert, you can deeply understand the user's instructions.

##Skills
* Analyze user instructions and treat a grab and a placement as an action
* Have strong logical ability
* When putting an object back to its original position, you need to consider the current relationship of the objects and then decide the order of transportation
* Only objects that have been moved can be put back to their original positions
* The directions are generally left, right, top, front, and back. If no value is specified, when left, it is [0.0,0.06], when right, it is [0.0,-0.06], when current, it is [0.06,0.0], when back, it is [-0.06,0.0], and when top, it is [0.0,0.0]

##Requirements and Restrictions
1. According to the input content, find the corresponding instructions in the function library and output the corresponding instructions.
2. Multiple actions need to be put in a list
3. Transportation is divided into two stages: picking up and putting down
4. Directly output data in json format, do not analyze, and do not output redundant content.
5. When restoring, if there is a move action in the instruction, then the move instruction needs to be returned, otherwise it returns ""
6. If the user asks for recovery/restore/put back in the same utterance as a pick-and-place (e.g. "put A on B and restore"), you MUST prepend [["remember"]] as the first action so initial poses are recorded, even if the user did not say "remember".
7. Format: {"action": "xx"}

##Available functions and operations:
* Pick up an object: [["Pick up the red object", "pick"]]
* Put it in the specific position of the specified object: [["Put it on the blue", "place", [0.0, 0.0]]]
* Remember the current object position: [["remember"]]
* Put the specified object back to the initial position: [["Put the blue back to the initial position", "restore"]]
* Put an object in the specific position of another object: [["Put the blue on the red", "pick_and_place", [0.0, 0.0]]]
* Recovery: [["", "recovery"]]

##Example:
Input: Put blue on yellow, put green on blue, put the object back to the initial position, only the steps of putting it back are needed
Output: {"action": [["Put green back to its original position", "restore"], ["Put blue back to its original position", "restore"]]}
Input: Put blue on red, put the object back to its original position, only need to put it back
Output: {"action": [["Put red back to its original position", "restore"]]}
Input: Remember, put blue to the right of red, restore
Output: {"action": [["remember"], ["Put blue to the right of red", "pick_and_place", [0.0, -0.06]], ["Put blue to the right of red", "recovery"]]}
Input: Remember, put blue 8cm to the left of yellow, put green 7cm to the right of blue, restore
Output: {"action": [["remember"], ["Put blue 8cm to the left of yellow", "pick_and_place", [0.0, 0.08]], ["Put green 7cm to the right of blue", "pick_and_place", [0.0, -0.07]],["Put blue 8cm to the left of yellow, and green 7cm to the right of blue", "recovery"]]}
Input: Put blue on red
Output: {"action": [["Put blue on red", "pick_and_place", [0.0, 0.0]]]}
Input: Pick up blue and put it on green
Output: {"action": [["Put blue on green", "pick_and_place", [0.0, 0.0]]]}
Input: Pick up red
Output: {"action": [["Pick up red", "pick"]]}
Input: Put on red
Output: {"action": [["Put on red", "place", [0.0, 0.0]]]}
Input: Put the red block on the green block and restore
Output: {"action": [["remember"], ["Put the red block on the green block", "pick_and_place", [0.0, 0.0]], ["Put the red block on the green block", "recovery"]]}
    '''
    
    VLLM_PROMPT1 = '''
**Role
You are an intelligent housekeeper robot capable of identifying and grasping various objects. Your task is to perform object detection and precise localization based on the image provided by the user, and return results in the specified Output Format.

**Task
1. Understand User Instruction
You will receive a sentence. Extract the mentioned object name from it.
The object's name must be expressed in English
Do not output any object not explicitly mentioned

2. Understand the Image
You will receive an image. Find the pixel coordinates of the top-left and bottom-right corners of the object matching the extracted name.
Do not detect or include unmentioned objects

**Special Notes
If the instruction refers only to the initial position, then place.object is "" and place.xyxy is []
If pick or place has no corresponding object, then "object" should be "" and "xyxy" should be []

**Output Format (Only return the JSON content below with no extra text, the coordinates (xmin, ymin, xmax, ymax) must be normalized to the range [0, 1]):
{
  "pick": {"object": name, "xyxy": [xmin, ymin, xmax, ymax]},
  "place": {"object": name, "xyxy": [xmin, ymin, xmax, ymax]
  "response": "A human-like reply, short and related to your next action"

**Example
Input: put it on the red one
Output:
{
  "pick": {"object": "", "xyxy": []},
  "place": {"object": "red", "xyxy": [xmin, ymin, xmax, ymax]},
  "response": "Got it. Placing the object on the red one now."
}
    '''

    VLLM_PROMPT2 = '''
**Role
You are an image recognition expert. Your task is to perform accurate object detection and localization based on the image provided by the user, and return the result in the specified Output Format.

**Task
1. Image Understanding
You will receive an image. If it contains red, green, and blue square blocks, identify and locate all of them in the image.

2. Object Localization - position
For each detected object, determine its location by returning the pixel coordinates of the top-left and bottom-right corners in the format [xmin, ymin, xmax, ymax].
Use your knowledge and visual recognition ability to determine a descriptive and specific English name for each object.

**Special Notes
Output all detected objects, do not miss any.
Use English names that highlight the object's characteristics.

**Output Format (Only output in the format below. Do not include any additional text, the coordinates (xmin, ymin, xmax, ymax) must be normalized to the range [0, 1])
{
  "object_name": [xmin, ymin, xmax, ymax],
  "object_name": [xmin, ymin, xmax, ymax]
}
    '''

# agent_process 与 object_transport 使用同一相机话题，常见分辨率为 640x480
VLLM_IMAGE_WIDTH = 640
VLLM_IMAGE_HEIGHT = 480

_VALID_ACTION_KEYS = ('pick', 'place', 'remember', 'recovery', 'restore', 'pick_and_place')

COMMAND_SERVICE_NAME = '/vllm_object_transport/command'


def parse_command_text(raw):
    raw = (raw or '').strip()
    if not raw:
        raise ValueError('payload is empty')
    if not (raw.startswith('{') or raw.startswith('[') or raw.startswith('"')):
        return raw
    data = json.loads(raw)
    if isinstance(data, str):
        text = data.strip()
        if not text:
            raise ValueError('payload text is empty')
        return text
    if isinstance(data, dict):
        text = data.get('instruction') or data.get('text') or data.get('query')
        if not text:
            raise ValueError('payload missing instruction/text/query')
        return str(text).strip()
    raise ValueError('unsupported payload type')


class VLLMObjectTransport(Node):
    @staticmethod
    def _resolve_calibration_config_dir():
        try:
            return os.path.join(get_package_share_directory('calibration'), 'config')
        except Exception:
            return '/home/ubuntu/ros2_ws/src/calibration/config'

    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        self.llm_result = ''
        self.running = True
        self.transport_finished = True
        self.action_finish = False
        self.play_audio_finish = False
        self.command_service_name = COMMAND_SERVICE_NAME
        self._command_busy = False
        self._node_ready = False
        self.lock = threading.RLock()
        self.create_service(Trigger, '~/get_node_status', self.get_node_status_callback)
        self.client = speech.OpenAIAPI(api_key, base_url)
        self.language = os.environ["ASR_LANGUAGE"]
        self.image_width = VLLM_IMAGE_WIDTH
        self.image_height = VLLM_IMAGE_HEIGHT
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        self.declare_parameter('camera_topic', '/depth_cam/rgb0/image_raw')
        self.declare_parameter('camera_info_topic', '/depth_cam/rgb0/camera_info')

        self.declare_parameter('mask_vllm_outside_work_area', True)
        # 工作区外填充色 BGR，默认纯黑
        self.declare_parameter('vllm_outside_mask_color_bgr', [0, 0, 0])

        self._calibration_config_dir = self._resolve_calibration_config_dir()
        self._work_corners = None
        self._work_extrinsic = None
        self._latest_camera_info = None
        camera_topic = self.get_parameter('camera_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self._camera_cb_group = ReentrantCallbackGroup()
        self._rgb_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            Image, camera_topic, self._image_callback,
            self._rgb_qos, callback_group=self._camera_cb_group)
        self.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_callback, self._rgb_qos)
        self._camera_info_ready = threading.Event()
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.asr_pub = self.create_publisher(String, '/vocal_detect/asr_result', 1)
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)
        self.create_subscription(Bool, 'object_transport/transport_finished', self.transport_finished_callback, 1)
        self.create_subscription(
            Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1,
            callback_group=self._camera_cb_group)

        self.get_logger().info("Waiting for service 'vocal_detect/enable_wakeup'...")
        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        self.awake_client.wait_for_service()
        self.get_logger().info("Service 'vocal_detect/enable_wakeup' found.")

        self.get_logger().info("Waiting for service 'agent_process/set_model'...")
        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_model_client.wait_for_service()
        self.get_logger().info("Service 'agent_process/set_model' found.")


        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        self.set_prompt_client.wait_for_service()

        self.set_vllm_content_client = self.create_client(SetContent, 'agent_process/set_vllm_content')
        self.set_vllm_content_client.wait_for_service()

        self.record_chat_client = self.create_client(SetBool, 'agent_process/record_chat')
        self.record_chat_client.wait_for_service()

        self.get_chat_client = self.create_client(Trigger, 'agent_process/get_chat')
        self.get_chat_client.wait_for_service()

        self.clear_chat_client = self.create_client(Empty, 'agent_process/clear_chat')
        self.clear_chat_client.wait_for_service()

        self.enter_client = self.create_client(Trigger, '/object_transport/enter')
        self.enter_client.wait_for_service()

        self.start_client = self.create_client(SetBool, '/object_transport/enable_transport')
        self.start_client.wait_for_service()

        self.set_pick_client = self.create_client(SetBox, 'object_transport/set_pick_position')
        self.set_pick_client.wait_for_service()

        self.set_place_client = self.create_client(SetBox, 'object_transport/set_place_position')
        self.set_place_client.wait_for_service()

        self.record_position_client = self.create_client(SetBox, 'object_transport/record_position')
        self.record_position_client.wait_for_service()

        self.clear_record_position_client = self.create_client(
            Empty, 'object_transport/clear_record_position')
        self.clear_record_position_client.wait_for_service()

        self.abort_transport_client = self.create_client(Trigger, '/object_transport/abort')
        self.abort_transport_client.wait_for_service()

        self._timer_cb_group = ReentrantCallbackGroup()
        self.command_srv = self.create_service(
            SetString,
            self.command_service_name,
            self.command_callback,
            callback_group=self._timer_cb_group,
        )
        self.timer = self.create_timer(0.0, self.init_process, callback_group=self._timer_cb_group)

    def _image_callback(self, ros_image):
        try:
            encoding = ros_image.encoding or 'bgr8'
            if encoding == 'rgb8':
                cv_image = self.bridge.imgmsg_to_cv2(ros_image, 'rgb8')
                bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = self.bridge.imgmsg_to_cv2(ros_image, 'bgr8')
        except Exception as e:
            self.get_logger().warning(f'VLLM 相机帧解码失败: {e}')
            return
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(np.array(bgr_image, dtype=np.uint8))

    def _camera_info_callback(self, msg):
        if msg.width > 0 and msg.height > 0:
            if (self.image_width, self.image_height) != (msg.width, msg.height):
                self.get_logger().info(
                    f'VLLM 图像尺寸: {msg.width}x{msg.height}（用于 0~1000 框转像素）')
            self.image_width = msg.width
            self.image_height = msg.height
            self._camera_info_ready.set()
        with self.lock:
            self._latest_camera_info = msg

    def _wait_camera_info(self, timeout_sec=5.0):
        if self._camera_info_ready.wait(timeout_sec):
            return True
        self.get_logger().warning(
            f'camera_info 未就绪，暂用默认 {self.image_width}x{self.image_height} 转换 VLLM 框')
        return False

    def _get_latest_image(self, timeout_sec=3.0):
        try:
            return self.image_queue.get(block=True, timeout=timeout_sec)
        except queue.Empty:
            return None

    def _try_load_work_area_calibration(self):
        if self._work_corners is not None and self._work_extrinsic is not None:
            return
        path = os.path.join(self._calibration_config_dir, 'transform.yaml')
        if not os.path.isfile(path):
            self.get_logger().debug(
                f'未找到 transform.yaml，VLLM 图像不做工作区外遮罩: {path}')
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self._work_corners = np.asarray(config['corners'], dtype=np.float64)
            self._work_extrinsic = np.asarray(config['extristric'], dtype=np.float64)
            self.get_logger().info(
                f'已加载 VLLM 工作区遮罩标定: corners={self._work_corners.shape}')
        except Exception as exc:
            self.get_logger().warning(f'加载工作区标定失败，跳过 VLLM 遮罩: {exc}')
            self._work_corners = None
            self._work_extrinsic = None

    def _mask_bgr_outside_work_area(self, bgr_image):
        if not self.get_parameter('mask_vllm_outside_work_area').value:
            return bgr_image
        self._try_load_work_area_calibration()
        if self._work_corners is None or self._work_extrinsic is None:
            return bgr_image
        with self.lock:
            cam_info = self._latest_camera_info
        if cam_info is None or cam_info.width <= 0:
            return bgr_image
        try:
            rgb_pts = utils.project_work_area_corners(
                self._work_corners,
                self._work_extrinsic,
                cam_info,
                plane_height=utils.WORK_AREA_PLANE_HEIGHT,
            )
        except Exception as exc:
            self.get_logger().warning(f'工作区投影失败，VLLM 图像不遮罩: {exc}')
            return bgr_image
        h, w = bgr_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.asarray(rgb_pts, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 255)
        out = np.array(bgr_image, copy=True)
        color = list(self.get_parameter('vllm_outside_mask_color_bgr').value)
        if len(color) >= 3:
            bgr = (int(color[0]), int(color[1]), int(color[2]))
        else:
            bgr = (0, 0, 0)
        out[mask == 0] = bgr
        return out

    def _attach_vllm_image(self, msg):
        bgr_image = self._get_latest_image()
        if bgr_image is None:
            return False
        bgr_image = self._mask_bgr_outside_work_area(bgr_image)
        msg.image = self.bridge.cv2_to_imgmsg(bgr_image, encoding='bgr8')
        return True

    def _parse_vllm_json(self, message):
        if message.startswith("```") and message.endswith("```"):
            message = message.strip("```").replace("json\n", "").strip()
        json_match = re.search(r'\{.*\}', message, re.DOTALL)
        if json_match:
            message = json_match.group()
        return json.loads(message)

    @staticmethod
    def _is_valid_action_item(action):
        if not isinstance(action, (list, tuple)) or len(action) < 1:
            return False
        return any(k in action for k in _VALID_ACTION_KEYS)

    def _parse_llm_action_list(self, llm_data):
        """解析 LLM 动作列表；无效/空/无关指令返回 None（对齐 llm_object_sorting 的 parse 逻辑）。"""
        if not llm_data or not str(llm_data).strip():
            return None
        text = llm_data.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.strip("```").replace("json\n", "").strip()
        if 'action' not in text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                return None
            try:
                payload = json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
        actions = payload.get('action')
        if actions in (None, '', []):
            return None
        if not isinstance(actions, list):
            return None
        valid = [a for a in actions if self._is_valid_action_item(a)]
        return valid if valid else None

    def _abort_object_transport(self):
        try:
            self.send_request(self.abort_transport_client, Trigger.Request())
        except Exception as e:
            self.get_logger().warning(f'中止 object_transport 失败: {e}')

    def _finish_interaction(self, tts_text):
        """与 llm_object_sorting / vllm_with_camera 一致：先 TTS，播完再 enable_wakeup。"""
        self._abort_object_transport()
        with self.lock:
            self.transport_finished = True
        msg = String()
        msg.data = tts_text
        self.tts_text_pub.publish(msg)
        self.action_finish = True
        self.get_logger().info(f'等待 TTS 结束后重新唤醒: {tts_text}')

    def _handle_invalid_command(self, llm_data=''):
        if self.language == 'Chinese':
            hint = '没有听清或与搬运无关，请重新唤醒后再说。'
        else:
            hint = 'Command not recognized. Please wake me up and try again.'
        self.get_logger().info(f'无效指令，跳过执行: {llm_data[:80]!r}')
        self._finish_interaction(hint)

    def _enable_wakeup_now(self):
        msg = SetBool.Request()
        msg.data = True
        self.send_request(self.awake_client, msg)
        self.get_logger().info('已重新开启语音唤醒')

    def _wait_transport_finished_cycles(self, count, timeout_sec=240.0):
        """等待 object_transport 发布 transport_finished（每完成一次 pick/place/restore 一次）。"""
        if count <= 0:
            return
        # 清空队列空闲时误发的 True，避免把「上一次空闲」当成本次搬运完成
        with self.lock:
            self.transport_finished = False
        deadline = time.time() + timeout_sec
        for i in range(count):
            while rclpy.ok() and time.time() < deadline:
                with self.lock:
                    if self.transport_finished:
                        break
                time.sleep(0.02)
            else:
                self.get_logger().error(
                    f'等待 object_transport 完成超时 (第 {i + 1}/{count} 次)')
                break
            if i < count - 1:
                with self.lock:
                    self.transport_finished = False

    def transport_finished_callback(self, msg):
        self.transport_finished = msg.data

    def play_audio_finish_callback(self, msg):
        self.play_audio_finish = msg.data

    def _vllm_xyxy_to_pixels(self, xyxy):
        if not xyxy or xyxy == "":
            return []
        if isinstance(xyxy, str):
            return []
        coords = [float(x) for x in xyxy[:4]]
        if len(coords) < 4:
            return []
        self._wait_camera_info(timeout_sec=0.1)
        w, h = self.image_width, self.image_height
        max_coord = max(coords)
        if max_coord <= 1.0:
            return [
                int(np.clip(coords[0] * w, 0, w - 1)),
                int(np.clip(coords[1] * h, 0, h - 1)),
                int(np.clip(coords[2] * w, 0, w - 1)),
                int(np.clip(coords[3] * h, 0, h - 1)),
            ]
        if max_coord <= 1000.0:
            return [
                int(np.clip(coords[0] * w / 1000.0, 0, w - 1)),
                int(np.clip(coords[1] * h / 1000.0, 0, h - 1)),
                int(np.clip(coords[2] * w / 1000.0, 0, w - 1)),
                int(np.clip(coords[3] * h / 1000.0, 0, h - 1)),
            ]
        return [
            int(np.clip(coords[0], 0, w - 1)),
            int(np.clip(coords[1], 0, h - 1)),
            int(np.clip(coords[2], 0, w - 1)),
            int(np.clip(coords[3], 0, h - 1)),
        ]

    def _remember_position_entries(self, initial_position):
        for key, vals in initial_position.items():
            if not isinstance(vals, (list, tuple)) or len(vals) < 4:
                continue
            if len(vals) == 4 and all(isinstance(v, (int, float)) for v in vals):
                label, xyxy = key, vals
            else:
                label = vals[0]
                xyxy = vals[1:5]
            yield label, xyxy

    def get_node_state(self, request, response):
        return response

    def get_node_status_callback(self, request, response):
        self.get_logger().info('[external_query] ~/get_node_status')
        if not self._node_ready:
            response.success = False
            response.message = 'initializing'
            return response
        with self.lock:
            busy = self._command_busy or bool(self.llm_result)
        if busy:
            response.success = True
            response.message = 'busy'
        else:
            response.success = True
            response.message = 'ready'
        return response

    def init_process(self):
        self.timer.cancel()

        msg = SetModel.Request()
        msg.model_type = 'llm'
        msg.model = llm_model
        msg.api_key = api_key
        msg.base_url = base_url
        self.send_request(self.set_model_client, msg)

        msg = SetString.Request()
        msg.data = LLM_PROMPT
        self.send_request(self.set_prompt_client, msg)
        
        msg = Empty.Request()
        self.send_request(self.clear_chat_client, msg)

        init_finish = self.create_client(Empty, 'object_transport/init_finish')
        init_finish.wait_for_service()
        self.get_logger().info('object_transport 已就绪，调用 enter...')
        self.send_request(self.enter_client, Trigger.Request())
        threading.Thread(target=lambda: speech.play_audio(start_audio_path), daemon=True).start()
        threading.Thread(target=self.process, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self._node_ready = True
        self.get_logger().info('\033[1;32m%s\033[0m' % 'vllm_object_transport start')
        self.get_logger().info('\033[1;32m%s\033[0m' % LLM_PROMPT)
        # self.get_logger().info(
        #     f'文本指令服务: {self.command_service_name} '
        #     f'(转发至 /vocal_detect/asr_result → agent_process)')

    def command_callback(self, request, response):
        self.get_logger().info('[external_query] /vllm_object_transport/command')
        try:
            text = parse_command_text(request.data)
        except Exception as exc:
            response.success = False
            response.message = f'invalid_payload: {exc}'
            return response
        with self.lock:
            if self._command_busy:
                response.success = False
                response.message = 'busy: previous command still processing'
                return response
            self._command_busy = True
            self.llm_result = ''
        self.get_logger().info(f'command (LLM direct): {text}')
        try:
            llm_answer = self.client.llm(text, LLM_PROMPT, model=llm_model)
        except Exception as exc:
            with self.lock:
                self._command_busy = False
            response.success = False
            response.message = f'llm_failed: {exc}'
            return response
        with self.lock:
            self.llm_result = llm_answer if llm_answer else ''
        response.success = True
        response.message = f'accepted: {text}'
        return response

    def send_request(self, client, msg, timeout_sec=60.0):
        future = client.call_async(msg)
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            if future.done():
                return future.result()
            time.sleep(0.01)
        raise TimeoutError(f'service call timeout: {client.srv_name}')

    def llm_result_callback(self, msg):
        with self.lock:
            self.llm_result = msg.data

    @staticmethod
    def _action_list_has_recovery(actions):
        return any(
            isinstance(a, (list, tuple)) and 'recovery' in a
            for a in actions
        )

    @staticmethod
    def _action_list_is_restore_only(actions):
        if not actions:
            return False
        for a in actions:
            if not isinstance(a, (list, tuple)) or 'restore' not in a:
                return False
        return True

    def _clear_record_positions(self):
        try:
            self.send_request(self.clear_record_position_client, Empty.Request())
            self.get_logger().info('已请求清空 object_transport 初态记录')
        except Exception as exc:
            self.get_logger().warning(f'清空 record_position 失败: {exc}')

    def _prepare_actions_for_recovery(self, actions):
        """含复原流程的新一轮指令：清空旧初态并保证先 remember（recovery 第二轮仅 restore 不重记）。"""
        if not actions:
            return actions
        if self._action_list_is_restore_only(actions):
            return list(actions)
        if not self._action_list_has_recovery(actions):
            return list(actions)
        self._clear_record_positions()
        rest = [
            a for a in actions
            if not (isinstance(a, (list, tuple)) and 'remember' in a)
        ]
        prepared = [['remember']] + rest
        self.get_logger().info(
            f'含复原：已清空旧记录并将 remember 置于队首 -> {prepared}')
        return prepared

    def remember_position(self):
        self._clear_record_positions()
        msg = SetContent.Request()
        if self.language == 'Chinese':
            msg.api_key = aliyun_api_key
            msg.base_url = aliyun_base_url
            msg.model = aliyun_vllm_model
        else:
            msg.api_key = vllm_api_key
            msg.base_url = vllm_base_url
            msg.model = vllm_model
        msg.prompt = VLLM_PROMPT2
        if not self._attach_vllm_image(msg):
            self.get_logger().error('无相机图像，无法记录初始位置')
            self._handle_invalid_command('无相机图像')
            return
        vllm_result = self.send_request(self.set_vllm_content_client, msg)
        initial_position = self._parse_vllm_json(vllm_result.message)
        n_record = 0
        for label, xyxy in self._remember_position_entries(initial_position):
            box = self._vllm_xyxy_to_pixels(xyxy)
            if not box:
                continue
            msg = SetBox.Request()
            msg.label = label
            msg.box = box
            self.send_request(self.record_position_client, msg)
            n_record += 1
        msg = SetBool.Request()
        msg.data = True
        self.send_request(self.start_client, msg)

        msg = SetBool.Request()
        msg.data = True
        self.send_request(self.record_chat_client, msg)
        # 给 object_transport 逐条处理 record 的时间，减少紧接着 pick 时帧不同步
        if n_record > 0:
            time.sleep(min(2.0, 0.35 + 0.12 * n_record))

    def recovery(self, text):
        if not text:
            msg = Trigger.Request()
            res = self.send_request(self.get_chat_client, msg)
            # self.get_logger().info(str(res.message)) 
            text = res.message.rsplit(",", 1)[0] if "," in res.message else res.message
            msg = SetBool.Request()
            msg.data = False
            self.send_request(self.record_chat_client, msg)
            msg = Empty.Request()
            self.send_request(self.clear_chat_client, msg)
        msg = String()
        if self.language == 'Chinese':
            msg.data = '{}, 只需要放回去的步骤'.format(text)
        else:
            msg.data = '{}, only the steps for putting it back'.format(text)
        self.get_logger().info('\033[1;32m%s\033[0m' % msg.data)
        self.asr_pub.publish(msg)

    def object_restore(self, query):
        msg = SetContent.Request()
        if self.language == 'Chinese':
            msg.api_key = aliyun_api_key
            msg.base_url = aliyun_base_url
            msg.model = aliyun_vllm_model
        else:
            msg.api_key = vllm_api_key
            msg.base_url = vllm_base_url
            msg.model = vllm_model
        msg.prompt = VLLM_PROMPT1
        msg.query = query
        if not self._attach_vllm_image(msg):
            self.get_logger().error('无相机图像，跳过 restore 视觉识别')
            self._handle_invalid_command('无相机图像')
            return 0
        vllm_result = self.send_request(self.set_vllm_content_client, msg)
        self.get_logger().info(f'vllm_result: {vllm_result}')
        position = self._parse_vllm_json(vllm_result.message)
        pick = position.get('pick') or {}
        place = position.get('place') or {}

        time.sleep(0.28)

        transport_ops = 0
        msg = SetBox.Request()
        msg.label = pick.get('object') or ''
        box = self._vllm_xyxy_to_pixels(pick.get('xyxy') or [])
        if box:
            msg.box = box
            msg.label = 'restore_pick'
            self.send_request(self.set_pick_client, msg)
            transport_ops += 1

        msg = SetBox.Request()
        place_label = (place.get('object') or '').strip()
        place_box = self._vllm_xyxy_to_pixels(place.get('xyxy') or [])
        if place_box:
            msg.label = place_label
            msg.box = place_box
        else:
            restore_label = place_label or (pick.get('object') or '').strip()
            if not restore_label:
                self.get_logger().error(
                    'restore：pick/place 均无有效物体名，无法下发 set_place')
                self._handle_invalid_command('restore 无目标物体')
                return 0
            msg.label = restore_label
            msg.box = []
        self.send_request(self.set_place_client, msg)
        transport_ops += 1

        # Enable sorting. 开启分拣
        msg = SetBool.Request()
        msg.data = True
        self.send_request(self.start_client, msg)
        self.transport_finished = False

        msg = Empty.Request()
        self.send_request(self.clear_chat_client, msg)

        msg = String()
        msg.data = position.get('response') or ''
        self.tts_text_pub.publish(msg)
        return transport_ops

    def get_object_position(self, query, offset=None, expect_pick_and_place=False):
        msg = SetContent.Request()
        if self.language == 'Chinese':
            msg.api_key = aliyun_api_key
            msg.base_url = aliyun_base_url
            msg.model = aliyun_vllm_model
        else:
            msg.api_key = vllm_api_key
            msg.base_url = vllm_base_url
            msg.model = vllm_model
        msg.prompt = VLLM_PROMPT1
        msg.query = query
        if not self._attach_vllm_image(msg):
            self.get_logger().error(
                '无相机图像，跳过 VLLM 识别（请检查 camera_topic 与相机节点）')
            self._handle_invalid_command('无相机图像')
            return False
        self._wait_camera_info()
        self.get_logger().info(
            f'调用 VLLM 视觉识别: {query} 图像={self.image_width}x{self.image_height}')
        vllm_result = self.send_request(self.set_vllm_content_client, msg)
        self.get_logger().info(f'vllm_result: {vllm_result}')
        position = self._parse_vllm_json(vllm_result.message)

        pick_box = self._vllm_xyxy_to_pixels(position['pick']['xyxy'])
        self.get_logger().info(f'pick xyxy(raw): {position["pick"]["xyxy"]} -> pixels: {pick_box}')
        place_box = self._vllm_xyxy_to_pixels(position['place']['xyxy'])
        if expect_pick_and_place:
            if not pick_box or not place_box:
                self.get_logger().error(
                    'pick_and_place 需要大模型同时给出有效 pick 与 place 框，当前结果不完整')
                self._handle_invalid_command('视觉识别结果不完整')
                return False
        # 等相机流略推进，使后续 FastSAM/深度与 VLLM 框更一致
        time.sleep(0.28)
        if pick_box:
            msg = SetBox.Request()
            msg.box = pick_box
            self.send_request(self.set_pick_client, msg)

        if place_box:
            msg = SetBox.Request()
            msg.box = place_box
            msg.offset = offset
            self.send_request(self.set_place_client, msg)

        # Enable sorting. 开启分拣
        msg = SetBool.Request()
        msg.data = True
        self.send_request(self.start_client, msg)
        self.transport_finished = False

        msg = String()
        msg.data = position.get('response') or ''
        self.tts_text_pub.publish(msg)
        return True

    def process(self):
        action_list = []
        while self.running:
            llm_data = ''
            with self.lock:
                if self.llm_result:
                    llm_data = self.llm_result
                    self.llm_result = ''

            if llm_data:
                actions = self._parse_llm_action_list(llm_data)
                if actions is None:
                    self._handle_invalid_command(llm_data)
                else:
                    action_list = self._prepare_actions_for_recovery(actions)
                    with self.lock:
                        self.transport_finished = True
                    self.get_logger().info(f'收到 LLM 动作: {action_list}')
            elif action_list:
                if self.transport_finished:
                    action = action_list[0]
                    self.get_logger().info(f'执行动作: {action}')
                    if 'remember' in action:
                        self.remember_position()
                    elif 'recovery' in action:
                        self.recovery(action[0])
                    elif 'pick' in action:
                        if self.get_object_position(action[0]):
                            self._wait_transport_finished_cycles(1)
                    elif 'place' in action:
                        if self.get_object_position(action[0], action[-1]):
                            self._wait_transport_finished_cycles(1)
                    elif 'pick_and_place' in action:
                        if self.get_object_position(
                                action[0], action[-1], expect_pick_and_place=True):
                            self._wait_transport_finished_cycles(2)
                    elif 'restore' in action:
                        n = self.object_restore(action[0])
                        if n:
                            self._wait_transport_finished_cycles(n)
                    else:
                        self._handle_invalid_command(str(action))
                        action_list.clear()
                        continue
                    del action_list[0]
                    if not action_list:
                        with self.lock:
                            self.transport_finished = True
                        self.action_finish = True
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.02)

            # 与 llm_object_sorting / vllm_with_camera：TTS 播完后再开启唤醒
            if self.play_audio_finish and self.action_finish:
                self.play_audio_finish = False
                self.action_finish = False
                with self.lock:
                    self._command_busy = False
                self._enable_wakeup_now()
        rclpy.shutdown()

def main():
    node = VLLMObjectTransport('vllm_object_transport')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()
