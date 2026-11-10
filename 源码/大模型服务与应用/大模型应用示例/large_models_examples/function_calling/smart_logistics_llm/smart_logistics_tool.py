#!/usr/bin/env python3
# encoding: utf-8
import ast
import json
import os
import queue
import re
import textwrap
import threading
import time

import rclpy
import numpy as np
from cv_bridge import CvBridge
from interfaces.srv import SetString as InterfaceSetString
from large_models.config import *
from large_models_msgs.msg import Tools
from large_models_msgs.srv import SetModel
from large_models_msgs.srv import SetString as LLMSetString
from large_models_msgs.srv import SetTools
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from speech import speech
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty, SetBool, Trigger


POINT_LABELS = {
    'Start': 'start point',
    'Point_A': 'area A',
    'Point_B': 'area B',
    'Point_C': 'area C',
    'Point_1': 'workstation 1',
    'Point_2': 'workstation 2',
    'Point_3': 'workstation 3',
    'End': 'end point',
}

COLOR_STATION_MAP = {
    'red': 'Point_1',
    'green': 'Point_2',
    'blue': 'Point_3',
}

AREA_SEQUENCE = ['Point_A', 'Point_B', 'Point_C']
VALID_POINTS = set(POINT_LABELS.keys())
VALID_COLORS = set(COLOR_STATION_MAP.keys())

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_available_locations',
            'description': '查询智能物流场景中机器人可导航的点位列表，包括A/B/C区、1/2/3号工位、起点和终点(Query the list of robot navigation locations in the smart logistics scene, including areas A/B/C, workstations 1/2/3, start, and end)',
            'parameters': {
                'type': 'object',
                'properties': {}
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_navigation_status',
            'description': '查询机器人当前导航状态，例如ready、moving或finish(Query the current robot navigation status, such as ready, moving, or finish)',
            'parameters': {
                'type': 'object',
                'properties': {}
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'describe_current_view',
            'description': '详细描述机器人摄像头当前看到的画面内容，以回答用户提出的具体问题，一般用在移动到某个地点之后(Describe in detail what the robot camera currently sees to answer user specific questions, usually after moving to a location)',
            'parameters': {
                'type': 'object',
                'properties': {
                    'question': {
                        'type': 'string',
                        'description': '关于当前画面的具体问题(Specific question about the current view)',
                    }
                },
                'required': ['question']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'navigate_to_point',
            'description': '导航到智能物流地图中的指定点位，A/B/C区用于观察和夹取，1/2/3号工位用于放置(Navigate to a specified point in the smart logistics map; areas A/B/C are for observation and picking, and workstations 1/2/3 are for placing)',
            'parameters': {
                'type': 'object',
                'properties': {
                    'destination': {
                        'type': 'string',
                        'description': '目标点位名称(Target point name)',
                        'enum': ['Start', 'Point_A', 'Point_B', 'Point_C', 'Point_1', 'Point_2', 'Point_3', 'End']
                    }
                },
                'required': ['destination']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'set_target_color',
            'description': '设置机械臂视觉夹取和放置时要识别的目标颜色(Set the target color recognized by the arm vision system for picking and placing)',
            'parameters': {
                'type': 'object',
                'properties': {
                    'color': {
                        'type': 'string',
                        'description': '目标工业元件颜色(Target industrial component color)',
                        'enum': ['red', 'green', 'blue']
                    }
                },
                'required': ['color']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'count_color',
            'description': '统计当前摄像头画面中红、绿、蓝工业元件/工件数量；如果指定颜色，会重点返回该颜色数量；工件和工业元件表示同一种目标(Count red, green, and blue industrial components/workpieces in the current camera view; if a color is specified, emphasize that color count; workpiece and industrial component mean the same target)',
            'parameters': {
                'type': 'object',
                'properties': {
                    'color': {
                        'type': 'string',
                        'description': '要重点统计的颜色(Color to count with priority)',
                        'enum': ['red', 'green', 'blue']
                    }
                }
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'pick_color_block',
            'description': '夹取当前区域中指定颜色的工业元件/工件，调用前应已导航到A/B/C区；工件和工业元件表示同一种目标(Pick the specified-color industrial component/workpiece in the current area; the robot should already be at area A/B/C before calling; workpiece and industrial component mean the same target)',
            'parameters': {
                'type': 'object',
                'properties': {
                    'color': {
                        'type': 'string',
                        'description': '要夹取的工业元件颜色(Industrial component color to pick)',
                        'enum': ['red', 'green', 'blue']
                    }
                },
                'required': ['color']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'place_color_block',
            'description': '在当前工位放置指定颜色的工业元件/工件，调用前应已导航到1/2/3号工位；工件和工业元件表示同一种目标(Place the specified-color industrial component/workpiece at the current workstation; the robot should already be at workstation 1/2/3 before calling; workpiece and industrial component mean the same target)',
            'parameters': {
                'type': 'object',
                'properties': {
                    'color': {
                        'type': 'string',
                        'description': '要放置的工业元件颜色(Industrial component color to place)',
                        'enum': ['red', 'green', 'blue']
                    }
                },
                'required': ['color']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_arm_state',
            'description': '获取机械臂当前状态，例如init、picking、pick_finish、placing或place_finish(Get the current arm state, such as init, picking, pick_finish, placing, or place_finish)',
            'parameters': {
                'type': 'object',
                'properties': {}
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'init_arm_pose',
            'description': '让机械臂回到智能物流初始姿态(Return the arm to the smart logistics initial pose)',
            'parameters': {
                'type': 'object',
                'properties': {}
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'run_transport_task',
            'description': '执行完整智能物流搬运任务：按任务序列从A/B/C区搜索指定颜色工业元件/工件，找到后夹取，再导航到对应工位放置；工件和工业元件表示同一种目标(Execute a complete smart logistics transport task: search areas A/B/C for specified-color industrial components/workpieces according to the task sequence, pick after finding one, then navigate to the corresponding workstation and place it; workpiece and industrial component mean the same target)',
            'parameters': {
                'type': 'object',
                'properties': {
                    'task_sequence': {
                        'type': 'string',
                        'description': 'JSON数组字符串，每项包含color，可选area、workstation，例如 [{"area":"Point_A","color":"red","workstation":"Point_1"}]；area表示搜索起点，不代表目标颜色一定在那里；如果该区域没有目标颜色，会继续按Point_A、Point_B、Point_C顺序往后搜索；area省略时从Point_A开始；workstation可省略，默认red->Point_1，green->Point_2，blue->Point_3(JSON array string; each item contains color and optional area/workstation, for example [{"area":"Point_A","color":"red","workstation":"Point_1"}]; area is the search starting point and does not guarantee the target color is there; if the target color is not in that area, continue searching through Point_A, Point_B, and Point_C in order; if area is omitted, start from Point_A; workstation can be omitted, defaulting to red->Point_1, green->Point_2, blue->Point_3)'
                    }
                },
                'required': ['task_sequence']
            }
        }
    },
]

language = os.environ.get('ASR_LANGUAGE', 'Chinese')
if language == 'Chinese':
    content_string = textwrap.dedent("""
    # 角色设定
    你是智能物流机器人助手，负责把用户的自然语言指令转换成工具调用，控制机器人完成A/B/C区观察、夹取和1/2/3号工位放置。

    # 场景点位
    - 起点/原点/出发点 = Start
    - A区 = Point_A，B区 = Point_B，C区 = Point_C
    - 1号工位 = Point_1，2号工位 = Point_2，3号工位 = Point_3
    - 终点 = End

    # 颜色
    工具参数里的颜色必须使用英文：红色=red，绿色=green，蓝色=blue。
    默认放置关系是：red放到Point_1，green放到Point_2，blue放到Point_3，除非用户明确指定其他工位。
    “工件”和“工业元件”表示同一种可夹取目标。

    # 工具使用规则
    1. 用户要求完整搬运任务时，优先调用run_transport_task，不要手动拆成多个低层工具。
    2. task_sequence必须是JSON数组字符串，例如：
       [{"area":"Point_A","color":"red","workstation":"Point_1"},{"area":"Point_B","color":"green","workstation":"Point_2"}]
    3. task_sequence中的area表示搜索起点，不代表该区域一定有目标颜色。比如“去A区工作台把红色工业元件送到1号工位”，area写Point_A，但如果A区没有red，工具会继续去Point_B、Point_C查找。
    4. 程序启动前机器人默认已经在Start，不需要在run_transport_task中额外传入起点。
    5. 任务完成后不需要自动回到Start或End；run_transport_task只导航到搜索区域和对应工位。
    6. 如果用户说明了A/B/C区对应的工件颜色，则按用户给出的区域和颜色生成 task_sequence。
    7. 如果用户只说明要寻找或搬运某种颜色，但没有明确该颜色在哪个区域，则导航查找顺序固定为Point_A、Point_B、Point_C。
    8. 如果用户只要求导航、计数、夹取、放置、机械臂状态，则调用对应单个工具。
    9. 调用工具前先用一句简短中文说明计划；工具完成后根据工具结果总结。
    10. 不要编造不存在的点位、颜色或工位。
    """)
else:
    content_string = textwrap.dedent("""
    You are a smart logistics robot assistant. Convert user requests into tool calls for navigation, color counting, picking, and placing.
    Use Start, Point_A, Point_B, Point_C, Point_1, Point_2, Point_3, and End as location parameters.
    Use color parameters red, green, or blue. The default placement map is red->Point_1, green->Point_2, blue->Point_3.
    "workpiece" and "industrial component" refer to the same pickable object.
    For a complete transport request, prefer run_transport_task with task_sequence as a JSON array string.
    If the user states which color is in each of Point_A, Point_B, and Point_C, build task_sequence from that mapping.
    In task_sequence, area is only the search starting point. If the requested color is not found there, continue searching through Point_A, Point_B, and Point_C in order.
    If the user asks to search for or transport a color but does not specify its area, start searching from Point_A, then Point_B, then Point_C.
    Before the program starts, the robot is assumed to already be at Start. Do not add a start location to run_transport_task.
    After the task is complete, do not automatically return to Start or End. run_transport_task only moves to search areas and target workstations.
    Give a short normal text plan before tool calls and summarize after tool results.
    """)


class LogColors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'


class SmartLogisticsTool(Node):
    def __init__(self, name):
        super().__init__(name)
        self.cb_group = ReentrantCallbackGroup()
        self.language = os.environ.get('ASR_LANGUAGE', 'Chinese')
        self.tool_queue = queue.Queue()
        self.image_queue = queue.Queue(maxsize=2)
        self.bridge = CvBridge()
        self.running = True
        if self.language == 'Chinese':
            self.vllm_model_name = 'qwen-vl-max-latest'
            self.vllm_client = speech.OpenAIAPI(api_key, base_url)
        else:
            self.vllm_model_name = vllm_model
            self.vllm_client = speech.OpenAIAPI(vllm_api_key, vllm_base_url)

        self.navigation_timeout = self.declare_parameter('navigation_timeout', 120.0).value
        self.arm_action_timeout = self.declare_parameter('arm_action_timeout', 90.0).value
        self.camera_topic = self.declare_parameter('camera_topic', '/depth_cam/rgb0/image_raw').value

        self.set_tool_client = self.create_client(SetTools, 'agent_process/set_tool')
        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_prompt_client = self.create_client(LLMSetString, 'agent_process/set_prompt')
        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')

        self.nav_set_pose_client = self.create_client(InterfaceSetString, '/claw_navigation_manager/set_pose')
        self.nav_status_client = self.create_client(Trigger, '/claw_navigation_manager/navigation_status')
        self.nav_list_client = self.create_client(Trigger, '/claw_navigation_manager/list_positions')
        self.nav_mode_client = self.create_client(Trigger, '/claw_navigation_manager/get_navigation_mode')

        self.set_color_client = self.create_client(InterfaceSetString, '/smart_logistics/set_color')
        self.pick_client = self.create_client(Trigger, '/smart_logistics/pick')
        self.place_client = self.create_client(Trigger, '/smart_logistics/place')
        self.init_pose_client = self.create_client(Trigger, '/smart_logistics/init_pose')
        self.arm_state_client = self.create_client(Trigger, '/smart_logistics/get_arm_state')
        self.count_color_client = self.create_client(Trigger, '/smart_logistics/count_color')

        self.create_subscription(Tools, 'agent_process/tools', self.tools_callback, 1, callback_group=self.cb_group)
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=self.cb_group)
        self.create_subscription(Image, self.camera_topic, self.image_callback, 1, callback_group=self.cb_group)
        self.tools_result_pub = self.create_publisher(Tools, 'agent_process/tools_result', 1)
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)

        self.timer = self.create_timer(0.1, self.init_process, callback_group=self.cb_group)

    def init_process(self):
        self.timer.cancel()
        self.wait_for_agent_services()
        self.configure_agent()
        threading.Thread(target=self.process, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        try:
            speech.play_audio(start_audio_path)
        except Exception as e:
            self.get_logger().warn('play start audio failed: %s' % str(e))
        self.get_logger().info('\033[1;32msmart logistics llm tool start\033[0m')

    def wait_for_agent_services(self):
        services = [
            (self.set_model_client, 'agent_process/set_model'),
            (self.set_prompt_client, 'agent_process/set_prompt'),
            (self.set_tool_client, 'agent_process/set_tool'),
        ]
        for client, name in services:
            if not client.wait_for_service(timeout_sec=8.0):
                self.get_logger().warn('Service %s is not available yet' % name)

    def configure_agent(self):
        msg = SetModel.Request()
        msg.model_type = 'llm_tools'
        if self.language == 'English':
            msg.model = 'qwen/qwen3-max'
            msg.api_key = vllm_api_key
            msg.base_url = vllm_base_url
        else:
            msg.model = 'qwen3-max'
            msg.api_key = api_key
            msg.base_url = base_url
        self.call_service(self.set_model_client, msg, 'agent_process/set_model', timeout_sec=10.0)

        prompt_msg = LLMSetString.Request()
        prompt_msg.data = content_string
        self.call_service(self.set_prompt_client, prompt_msg, 'agent_process/set_prompt', timeout_sec=10.0)

        tools_msg = SetTools.Request()
        tools_msg.tools = [json.dumps(tool, ensure_ascii=False) for tool in tools]
        self.call_service(self.set_tool_client, tools_msg, 'agent_process/set_tool', timeout_sec=10.0)

    def get_node_state(self, request, response):
        return response

    def llm_result_callback(self, msg):
        self.get_logger().info('%s%sLLM Reply: %s%s' % (
            LogColors.YELLOW, LogColors.BOLD, msg.data, LogColors.RESET))
        if msg.data:
            tts_msg = String()
            tts_msg.data = msg.data
            self.tts_text_pub.publish(tts_msg)

    def play_audio_finish_callback(self, msg):
        if msg.data and self.awake_client.service_is_ready():
            awake_msg = SetBool.Request()
            awake_msg.data = True
            self.call_service(self.awake_client, awake_msg, 'vocal_detect/enable_wakeup', timeout_sec=2.0)

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, 'bgr8')
        bgr_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(bgr_image)

    def tools_callback(self, msg):
        self.get_logger().info('%s%sTool request: id=%s name=%s data=%s%s' % (
            LogColors.GREEN, LogColors.BOLD, msg.id, msg.name, msg.data, LogColors.RESET))
        try:
            args = json.loads(msg.data) if msg.data else {}
        except Exception:
            args = {}
        self.tool_queue.put((msg.id, msg.name, args))

    def process(self):
        while rclpy.ok() and self.running:
            try:
                tool_id, tool_name, args = self.tool_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = self.dispatch_tool(tool_name, args)
            except Exception as e:
                self.get_logger().error('Tool %s error: %s' % (tool_name, str(e)))
                result = 'Tool execution failed: %s' % str(e)

            self.tools_result_pub.publish(Tools(id=tool_id, name=tool_name, data=result))

    def dispatch_tool(self, tool_name, args):
        if tool_name == 'get_available_locations':
            return self.get_available_locations()
        if tool_name == 'get_navigation_status':
            return self.get_navigation_status()
        if tool_name == 'describe_current_view':
            return self.describe_current_view(args.get('question', 'What can you see now?'))
        if tool_name == 'navigate_to_point':
            return self.navigate_to_point(args.get('destination', ''))
        if tool_name == 'set_target_color':
            return self.set_target_color(args.get('color', ''))
        if tool_name == 'count_color':
            return self.count_color(args.get('color'))
        if tool_name == 'pick_color_block':
            return self.pick_color_block(args.get('color', ''))
        if tool_name == 'place_color_block':
            return self.place_color_block(args.get('color', ''))
        if tool_name == 'get_arm_state':
            return self.get_arm_state()
        if tool_name == 'init_arm_pose':
            return self.init_arm_pose()
        if tool_name == 'run_transport_task':
            return self.run_transport_task(
                args.get('task_sequence') or args.get('tasks') or ''
            )
        return 'Unknown tool: %s' % tool_name

    def call_service(self, client, request, service_name, timeout_sec=10.0):
        if not client.wait_for_service(timeout_sec=3.0):
            return False, 'Service is unavailable: %s' % service_name
        future = client.call_async(request)
        start = time.time()
        while rclpy.ok() and time.time() - start < timeout_sec:
            if future.done():
                try:
                    return True, future.result()
                except Exception as e:
                    return False, 'Service call failed for %s: %s' % (service_name, str(e))
            time.sleep(0.02)
        return False, 'Service call timed out for %s' % service_name

    def call_trigger(self, client, service_name, timeout_sec=10.0):
        ok, response = self.call_service(client, Trigger.Request(), service_name, timeout_sec)
        if not ok:
            return False, response
        return response.success, response.message

    def call_set_string(self, client, service_name, data, timeout_sec=10.0):
        request = InterfaceSetString.Request()
        request.data = data
        ok, response = self.call_service(client, request, service_name, timeout_sec)
        if not ok:
            return False, response
        return response.success, response.message

    def get_available_locations(self):
        ok, message = self.call_trigger(self.nav_list_client, '/claw_navigation_manager/list_positions', timeout_sec=5.0)
        static_locations = ', '.join(['%s(%s)' % (name, label) for name, label in POINT_LABELS.items()])
        if ok:
            return 'Available locations: %s. Navigation manager response: %s' % (static_locations, message)
        return 'Available locations: %s. Navigation manager query failed: %s' % (static_locations, message)

    def get_navigation_status(self):
        ok, message = self.call_trigger(self.nav_status_client, '/claw_navigation_manager/navigation_status', timeout_sec=5.0)
        if ok:
            return 'Current navigation status: %s' % message
        return 'Failed to query navigation status: %s' % message

    def describe_current_view(self, question):
        try:
            time.sleep(1.0)
            rgb_image = self.image_queue.get(block=True, timeout=5.0)
            if self.language == 'Chinese':
                prompt = textwrap.dedent("""
                请仔细观察机器人摄像头当前画面，并回答用户的问题。
                场景是智能物流工作台，工件和工业元件表示同一种可夹取目标。
                回答要简洁，不要反问，优先说明是否看到目标颜色工件、数量或位置。
                """)
            else:
                prompt = textwrap.dedent("""
                Observe the robot camera image carefully and answer the user question.
                The scene is a smart logistics workstation. "workpiece" and "industrial component" mean the same pickable object.
                Keep the answer concise. Do not ask follow-up questions. Prioritize whether target-color objects are visible, their count, or their position.
                """)
            description = self.vllm_client.vllm(question, rgb_image, prompt=prompt, model=self.vllm_model_name)
            self.get_logger().info('%s%sVLLM Reply: %s%s' % (
                LogColors.YELLOW, LogColors.BOLD, description, LogColors.RESET))
            tts_msg = String()
            tts_msg.data = description
            self.tts_text_pub.publish(tts_msg)
            return 'View description completed. Result: %s' % description
        except queue.Empty:
            return 'View description failed: no camera image was received'
        except Exception as e:
            self.get_logger().error('Describe current view error: %s' % str(e))
            return 'View description failed: %s' % str(e)

    def navigate_to_point(self, destination, wait=True):
        point = self.normalize_point(destination)
        if not point:
            return 'Navigation failed: unknown destination %s. Available locations: %s' % (
                destination, ', '.join(POINT_LABELS.keys()))

        ok, message = self.call_set_string(
            self.nav_set_pose_client,
            '/claw_navigation_manager/set_pose',
            point,
            timeout_sec=10.0
        )
        if not ok:
            return 'Navigation to %s (%s) failed: %s' % (point, POINT_LABELS[point], message)
        if not wait:
            return 'Navigation goal sent: %s (%s)' % (point, POINT_LABELS[point])

        reached, status = self.wait_for_navigation(point)
        if reached:
            return 'Arrived at %s (%s)' % (point, POINT_LABELS[point])
        return 'Navigation to %s (%s) was not confirmed: %s' % (point, POINT_LABELS[point], status)

    def wait_for_navigation(self, point):
        start = time.time()
        last_status = ''
        while rclpy.ok() and time.time() - start < self.navigation_timeout:
            ok, message = self.call_trigger(
                self.nav_status_client,
                '/claw_navigation_manager/navigation_status',
                timeout_sec=3.0
            )
            if ok:
                last_status = message
                if message == 'finish':
                    return True, message
            else:
                last_status = message
            time.sleep(0.5)
        return False, 'Timed out waiting to arrive at %s. Last status: %s' % (point, last_status)

    def set_target_color(self, color):
        target = self.normalize_color(color)
        if not target:
            return 'Set target color failed: unknown color %s. Available colors: red, green, blue' % color
        ok, message = self.call_set_string(
            self.set_color_client,
            '/smart_logistics/set_color',
            target,
            timeout_sec=5.0
        )
        if ok:
            return 'Target color set to %s' % target
        return 'Set target color to %s failed: %s' % (target, message)

    def count_color(self, color=None):
        target = self.normalize_color(color) if color else None
        if color and not target:
            return 'Color counting failed: unknown color %s' % color
        if target:
            set_result = self.set_target_color(target)
            if self.is_failure(set_result):
                return set_result

        ok, message = self.call_trigger(
            self.count_color_client,
            '/smart_logistics/count_color',
            timeout_sec=10.0
        )
        if not ok:
            return 'Color counting failed: %s' % message

        counts = self.parse_counts(message)
        if target and target in counts:
            return 'Current view has %d %s objects. Full count: %s' % (counts[target], target, message)
        return 'Current view color count: %s' % message

    def pick_color_block(self, color):
        target = self.normalize_color(color)
        if not target:
            return 'Pick failed: unknown color %s' % color
        set_result = self.set_target_color(target)
        if self.is_failure(set_result):
            return set_result
        ok, message = self.call_trigger(
            self.pick_client,
            '/smart_logistics/pick',
            timeout_sec=self.arm_action_timeout
        )
        if ok:
            return 'Finished picking the %s object: %s' % (target, message)
        return 'Picking the %s object failed: %s' % (target, message)

    def place_color_block(self, color):
        target = self.normalize_color(color)
        if not target:
            return 'Place failed: unknown color %s' % color
        set_result = self.set_target_color(target)
        if self.is_failure(set_result):
            return set_result
        ok, message = self.call_trigger(
            self.place_client,
            '/smart_logistics/place',
            timeout_sec=self.arm_action_timeout
        )
        if ok:
            return 'Finished placing the %s object: %s' % (target, message)
        return 'Placing the %s object failed: %s' % (target, message)

    def get_arm_state(self):
        ok, message = self.call_trigger(
            self.arm_state_client,
            '/smart_logistics/get_arm_state',
            timeout_sec=5.0
        )
        if ok:
            return 'Current arm state: %s' % message
        return 'Failed to get arm state: %s' % message

    def init_arm_pose(self):
        ok, message = self.call_trigger(
            self.init_pose_client,
            '/smart_logistics/init_pose',
            timeout_sec=self.arm_action_timeout
        )
        if ok:
            return 'Arm returned to the initial pose: %s' % message
        return 'Failed to return arm to the initial pose: %s' % message

    def run_transport_task(self, task_sequence):
        tasks = self.parse_task_sequence(task_sequence)
        if not tasks:
            return 'Task parsing failed: task_sequence must be a JSON array string containing color'

        summary = []
        for index, task in enumerate(tasks, start=1):
            start_area = self.normalize_point(self.first_value(task, ['area', 'source', 'from', 'location', 'point']))
            color = self.normalize_color(self.first_value(task, ['color', 'colour']))
            workstation = self.normalize_point(self.first_value(task, ['workstation', 'station', 'target', 'to', 'destination']))
            if color and not workstation:
                workstation = COLOR_STATION_MAP.get(color)

            if start_area and start_area not in AREA_SEQUENCE:
                summary.append('Task item %d failed: area must be Point_A, Point_B, or Point_C' % index)
                return self.join_summary(summary)
            if not color:
                summary.append('Task item %d failed: color must be red, green, or blue' % index)
                return self.join_summary(summary)
            if not workstation or workstation not in ['Point_1', 'Point_2', 'Point_3']:
                summary.append('Task item %d failed: workstation must be Point_1, Point_2, or Point_3' % index)
                return self.join_summary(summary)

            search_areas = self.get_search_areas(start_area)
            summary.append('Task item %d: search for %s from %s, then deliver to %s (%s)' % (
                index, color, ' -> '.join(search_areas), workstation, POINT_LABELS[workstation]))

            found_area = None
            for area in search_areas:
                result = self.navigate_to_point(area)
                summary.append(result)
                if self.is_failure(result):
                    return self.join_summary(summary)

                count_result = self.count_color(color)
                summary.append(count_result)
                if self.is_failure(count_result):
                    return self.join_summary(summary)
                counts = self.parse_counts(count_result)
                if color in counts and counts[color] > 0:
                    found_area = area
                    summary.append('Found %d %s object(s) at %s (%s)' % (
                        counts[color], color, area, POINT_LABELS[area]))
                    break
                summary.append('No %s object detected at %s (%s); continue searching' % (
                    color, area, POINT_LABELS[area]))

            if not found_area:
                summary.append('Task item %d stopped: no %s object was found in Point_A, Point_B, or Point_C' % (index, color))
                return self.join_summary(summary)

            result = self.pick_color_block(color)
            summary.append(result)
            if self.is_failure(result):
                return self.join_summary(summary)

            result = self.navigate_to_point(workstation)
            summary.append(result)
            if self.is_failure(result):
                return self.join_summary(summary)

            result = self.place_color_block(color)
            summary.append(result)
            if self.is_failure(result):
                return self.join_summary(summary)

        summary.append('Smart logistics transport task completed')
        return self.join_summary(summary)

    def normalize_point(self, value):
        if value is None:
            return None
        text = str(value).strip()
        if text in VALID_POINTS:
            return text
        normalized = text.replace('-', '_').lower()
        for point in VALID_POINTS:
            if normalized == point.lower():
                return point
        return None

    def normalize_color(self, value):
        if value is None:
            return None
        key = str(value).strip().lower()
        return key if key in VALID_COLORS else None

    def get_search_areas(self, start_area=None):
        if not start_area:
            return AREA_SEQUENCE[:]
        start_index = AREA_SEQUENCE.index(start_area)
        return AREA_SEQUENCE[start_index:]

    def parse_task_sequence(self, task_sequence):
        data = task_sequence
        if isinstance(data, str):
            data = data.strip()
            if not data:
                return []
            try:
                data = json.loads(data)
            except Exception:
                try:
                    data = ast.literal_eval(data)
                except Exception:
                    return []
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def parse_counts(self, message):
        counts = {}
        if not message:
            return counts
        for color_text, value_text in re.findall(r'(red|green|blue|红色|绿色|蓝色|红|绿|蓝)\s*[:：]\s*(-?\d+)', str(message), re.IGNORECASE):
            color = self.normalize_color(color_text)
            if color:
                counts[color] = int(value_text)
        if counts:
            return counts
        for part in str(message).replace('；', ';').replace('，', ',').replace(';', ',').split(','):
            if ':' not in part:
                continue
            key, value = part.split(':', 1)
            color = self.normalize_color(key)
            if not color:
                continue
            try:
                counts[color] = int(value.strip())
            except ValueError:
                continue
        return counts

    def first_value(self, data, keys):
        for key in keys:
            if key in data and data[key] not in [None, '']:
                return data[key]
        return None

    def is_failure(self, result):
        failure_words = [
            'failed',
            'failure',
            'timed out',
            'unavailable',
            'unknown',
            'not confirmed',
            'stopped',
        ]
        result = result.lower()
        return any(word in result for word in failure_words)

    def join_summary(self, items):
        return '\n'.join(items)


def main():
    rclpy.init()
    node = SmartLogisticsTool('smart_logistics_tool')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
