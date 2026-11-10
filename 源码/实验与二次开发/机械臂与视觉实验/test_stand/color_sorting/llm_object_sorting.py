#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2024/11/18
import os
import time
import json
import rclpy
import threading
from config import *
from speech import speech
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, SetBool, Empty

from large_models.config import *
from large_models_msgs.srv import SetModel, SetString as LLMSetString

from interfaces.srv import SetStringBool, SetString, GetBoxPosition
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

PICK_COLORS = ('red', 'green', 'blue')
PLACE_DIRECTIONS = ('left_up', 'left_down', 'right_up', 'right_down')

COLOR_ALIASES = {
    'red': 'red', '红': 'red', '红色': 'red',
    'green': 'green', '绿': 'green', '绿色': 'green',
    'blue': 'blue', '蓝': 'blue', '蓝色': 'blue',
}

PLACE_BOX_ALIASES = {
    'red': 'red', '红': 'red', '红色': 'red', '红色盒子': 'red', '红盒子': 'red',
    'green': 'green', '绿': 'green', '绿色': 'green', '绿色盒子': 'green', '绿盒子': 'green',
    'yellow': 'yellow', '黄': 'yellow', '黄色': 'yellow', '黄色盒子': 'yellow', '黄盒子': 'yellow',
    'blue': 'blue', '蓝': 'blue', '蓝色': 'blue', '蓝色盒子': 'blue', '蓝盒子': 'blue',
}

if os.environ["ASR_LANGUAGE"] == 'Chinese':
    PROMPT = '''
#角色任务
你是一款智能机械臂，需要根据用户的语音指令，生成夹取与放置任务。

##要求与限制
1. 夹取（pick）使用视觉+运动学抓取桌面上的彩色木块；放置（place）由机械臂动作组完成，你只输出要放入的颜色盒子。
2. 可夹取三种颜色木块：红色(red)、绿色(green)、蓝色(blue)。
3. 可放置到四种颜色盒子：红色(red)、绿色(green)、黄色(yellow)、蓝色(blue)。盒子实际方位由视觉标定，你只需输出目标盒子颜色。
4. 为任务编织一句精炼（10至30字）、风趣且变化无穷的反馈信息。
5. 直接输出 JSON，不要分析，不要输出多余内容。
6. 格式：{"pick": "颜色", "place": "盒子颜色", "response": "回复"}；多条指令时用 tasks 数组。

##结构要求
- "pick"：要夹取的颜色，取值 red / green / blue（可带“木块”等描述，但颜色必须明确）。
- "place"：目标颜色盒子，取值 red / green / yellow / blue（表示放进对应颜色的盒子里）。
- "tasks"：可选，多条任务时为 [{"pick":"red","place":"red"}, ...]，此时可省略顶层 pick/place。
- "response"：简短语音回复。
- 无法理解指令时 pick、place 置为 ""。

##任务示例
输入：夹取红色木块放到红色盒子里面
输出：{"pick": "red", "place": "red", "response": "收到，红色木块回红色家"}

输入：把蓝色木块放到红色盒子里面
输出：{"pick": "blue", "place": "red", "response": "蓝色木块去红色盒子报到"}

输入：夹取红色木块放到红色盒子，把蓝色木块放到蓝色盒子
输出：{"tasks": [{"pick": "red", "place": "red"}, {"pick": "blue", "place": "blue"}], "response": "红归红盒、蓝归蓝盒，马上开工"}
'''
else:
    PROMPT = '''
#Role
You are an intelligent robot arm. Generate pick-and-place tasks from user voice commands.

##Requirements
1. Pick uses vision + kinematics to grasp colored blocks; place uses action groups — output the target color box, not a direction.
2. Pick colors: red, green, blue.
3. Place into color boxes: red, green, yellow, blue — output red / green / yellow / blue for "place".
4. Add a concise (10–30 words), witty "response".
5. Output JSON only, no extra text.
6. Format: {"pick": "color", "place": "box_color", "response": "..."}; use a "tasks" array for multiple commands.

##Fields
- "pick": red / green / blue
- "place": red / green / yellow / blue (target color box)
- "tasks": optional list of {"pick","place"} objects
- "response": short spoken reply
- Use empty strings for pick/place when the command is not understood.

##Examples
Input: Pick up the red block and put it in the red box
Output: {"pick": "red", "place": "red", "response": "Red block heading to the red box"}

Input: Put the blue block in the red box
Output: {"pick": "blue", "place": "red", "response": "Blue block to the red box"}

Input: Put red in the red box and blue in the blue box
Output: {"tasks": [{"pick": "red", "place": "red"}, {"pick": "blue", "place": "blue"}], "response": "Red to red, blue to blue — on it"}
'''


def parse_llm_json(text):
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def normalize_pick(pick):
    if not pick:
        return None
    raw = str(pick).strip()
    text = raw.lower()
    for key, color in COLOR_ALIASES.items():
        if key in raw or key in text:
            return color
    return text if text in PICK_COLORS else None


def normalize_place_box(place):
    if not place:
        return None
    raw = str(place).strip()
    text = raw.lower()
    for key, color in PLACE_BOX_ALIASES.items():
        if key in raw or key in text:
            return color
    if text in ('red', 'green', 'yellow', 'blue'):
        return text
    return None


def extract_tasks(result):
    tasks = []
    if isinstance(result.get('tasks'), list):
        for item in result['tasks']:
            if not isinstance(item, dict):
                continue
            pick = normalize_pick(item.get('pick', ''))
            place_box = normalize_place_box(item.get('place', ''))
            if pick and place_box:
                tasks.append({'pick': pick, 'place_box': place_box})
    pick = normalize_pick(result.get('pick', ''))
    place_box = normalize_place_box(result.get('place', ''))
    if pick and place_box:
        tasks.append({'pick': pick, 'place_box': place_box})
    return tasks


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
        text = (
            data.get('instruction')
            or data.get('text')
            or data.get('query')
        )
        if not text:
            raise ValueError('payload missing instruction/text/query')
        return str(text).strip()

    raise ValueError('unsupported payload type')


class LLMColorSorting(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)

        self.llm_result = ''
        self.action_finish = False
        self.play_audio_finish = False
        self.sorting_done = False
        self.running = True
        self._node_ready = False
        self.command_service_name = '/llm_color_sorting/command'
        self.create_service(Trigger, '~/get_node_status', self.get_node_status_callback)

        timer_cb_group = ReentrantCallbackGroup()
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)
        self.asr_pub = self.create_publisher(String, 'vocal_detect/asr_result', 1)
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=timer_cb_group)
        self.create_subscription(Bool, 'vocal_detect/wakeup', self.wakeup_callback, 1)
        self.create_subscription(Bool, 'object_sorting/sorting_done', self.sorting_done_callback, 1, callback_group=timer_cb_group)
        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        self.awake_client.wait_for_service()

        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_model_client.wait_for_service()
        self.set_prompt_client = self.create_client(LLMSetString, 'agent_process/set_prompt')
        self.set_prompt_client.wait_for_service()
        self.enter_client = self.create_client(Trigger, 'object_sorting/enter')
        self.enter_client.wait_for_service()
        self.start_client = self.create_client(SetBool, 'object_sorting/enable_sorting')
        self.start_client.wait_for_service()
        self.set_target_client = self.create_client(SetStringBool, 'object_sorting/set_target')
        self.set_target_client.wait_for_service()
        self.set_place_client = self.create_client(SetString, 'object_sorting/set_place')
        self.set_place_client.wait_for_service()
        self.get_box_position_client = self.create_client(
            GetBoxPosition, 'object_sorting/get_box_position')
        self.get_box_position_client.wait_for_service()
        self.red_box = ''
        self.green_box = ''
        self.yellow_box = ''
        self.blue_box = ''
        self._last_box_uncalib_log = 0.0
        self.command_srv = self.create_service(
            LLMSetString,
            self.command_service_name,
            self.command_callback,
            callback_group=timer_cb_group,
        )
        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def get_node_state(self, request, response):
        return response

    def get_node_status_callback(self, request, response):
        self.get_logger().info('[external_query] ~/get_node_status')
        if not self._node_ready:
            response.success = False
            response.message = 'initializing'
            return response
        if self.llm_result:
            response.success = True
            response.message = 'busy'
        else:
            response.success = True
            response.message = 'ready'
        return response

    def init_process(self):
        self.timer.cancel()

        msg = SetModel.Request()
        msg.model = llm_model
        msg.model_type = 'llm'
        msg.api_key = api_key
        msg.base_url = base_url
        self.send_request(self.set_model_client, msg)

        msg = LLMSetString.Request()
        msg.data = PROMPT
        self.send_request(self.set_prompt_client, msg)

        self.send_request(self.enter_client, Trigger.Request())
        threading.Thread(target=self._wait_and_load_box_params, daemon=True).start()
        speech.play_audio(start_audio_path)
        threading.Thread(target=self.process, daemon=True).start()
        self.create_service(Empty, '~/init_finish', self.get_node_state)
        self._node_ready = True
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')
        self.get_logger().info(
            f'command service ready: {self.command_service_name} (simulate ASR via vocal_detect/asr_result)'
        )
        self.get_logger().info('\033[1;32m%s\033[0m' % PROMPT)

    def command_callback(self, request, response):
        self.get_logger().info('[external_query] /llm_color_sorting/command')
        try:
            text = parse_command_text(request.data)
        except Exception as exc:
            response.success = False
            response.message = f'invalid_payload: {exc}'
            return response

        if self.llm_result:
            response.success = False
            response.message = 'busy: previous command still processing'
            return response

        msg = String()
        msg.data = text
        self.get_logger().info(f'command (simulate ASR): {text}')
        self.asr_pub.publish(msg)
        response.success = True
        response.message = f'accepted: {text}'
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def wakeup_callback(self, msg):
        if msg.data and self.llm_result:
            self.send_request(self.enter_client, Trigger.Request())

    def llm_result_callback(self, msg):
        self.llm_result = msg.data

    def play_audio_finish_callback(self, msg):
        self.play_audio_finish = msg.data

    def sorting_done_callback(self, msg):
        if msg.data:
            self.sorting_done = True

    def wait_sorting_done(self, timeout=120.0):
        self.sorting_done = False
        deadline = time.time() + timeout
        while self.running and rclpy.ok() and time.time() < deadline:
            if self.sorting_done:
                return True
            time.sleep(0.05)
        return False

    def _wait_and_load_box_params(self):
        deadline = time.time() + 120.0
        sleep_s = 1.0
        while self.running and rclpy.ok() and time.time() < deadline:
            if self.refresh_box_positions():
                return
            time.sleep(sleep_s)
            sleep_s = min(sleep_s * 1.5, 8.0)
        self.get_logger().warn('timeout waiting for object_sorting box parameters')

    def refresh_box_positions(self):
        if not self.get_box_position_client.service_is_ready():
            return False
        self.get_logger().info('[external_query] → object_sorting/get_box_position')
        req = GetBoxPosition.Request()
        res = self.send_request(self.get_box_position_client, req)
        if not res:
            self.get_logger().warn('get_box_position: no response')
            return False
        self.red_box = res.red_box or ''
        self.green_box = res.green_box or ''
        self.yellow_box = res.yellow_box or ''
        self.blue_box = res.blue_box or ''
        has = bool(self.red_box or self.green_box or self.yellow_box or self.blue_box)
        if has:
            self.get_logger().info(
                f'box params: red={self.red_box} green={self.green_box} '
                f'yellow={self.yellow_box} blue={self.blue_box}')
        else:
            now = time.time()
            if now - self._last_box_uncalib_log > 20.0:
                self._last_box_uncalib_log = now
                self.get_logger().warn(
                    'box positions not calibrated yet (message=%s); '
                    'wait for camera + auto detect or call /object_sorting/detect_box_position'
                    % getattr(res, 'message', ''))
        return has

    def get_place_box_direction(self, place_box_color):
        if place_box_color == 'red':
            return self.red_box
        if place_box_color == 'green':
            return self.green_box
        if place_box_color == 'yellow':
            return self.yellow_box
        if place_box_color == 'blue':
            return self.blue_box
        return ''

    def resolve_place_direction(self, place_box_color):
        direction = self.get_place_box_direction(place_box_color)
        if not direction:
            self.get_logger().warn(
                f'unknown position for {place_box_color} box, refreshing...')
            self.refresh_box_positions()
            direction = self.get_place_box_direction(place_box_color)
        return direction

    def set_pick_target(self, color):
        color_msg = SetStringBool.Request()
        for label in list(PICK_COLORS) + ['tag1', 'tag2', 'tag3']:
            color_msg.data_str = label
            color_msg.data_bool = label == color
            self.send_request(self.set_target_client, color_msg)

    def run_pick_place_task(self, pick, place_box):
        place_direction = self.resolve_place_direction(place_box)
        if not place_direction:
            self.get_logger().warn(f'no direction for {place_box} color box')
            return False
        place_msg = SetString.Request()
        place_msg.data = place_direction
        place_res = self.send_request(self.set_place_client, place_msg)
        if not place_res.success:
            self.get_logger().warn(
                f'invalid place direction: {place_direction} ({place_box} box)')
            return False
        self.set_pick_target(pick)
        self.get_logger().info(
            f'pick={pick}, place_box={place_box} -> {place_direction}')
        start_msg = SetBool.Request()
        start_msg.data = True
        self.send_request(self.start_client, start_msg)
        return self.wait_sorting_done()

    def process(self):
        while self.running:
            if self.llm_result:
                msg = String()
                result = parse_llm_json(self.llm_result)
                if result and ('pick' in result or 'tasks' in result):
                    tasks = extract_tasks(result)
                    if tasks:
                        for task in tasks:
                            if not self.run_pick_place_task(
                                    task['pick'], task['place_box']):
                                self.get_logger().warn(
                                    f"task failed or timeout: {task['pick']} -> {task['place_box']} box")
                            stop_msg = SetBool.Request()
                            stop_msg.data = False
                            self.send_request(self.start_client, stop_msg)
                            time.sleep(0.3)
                    if 'response' in result:
                        msg.data = result['response']
                    elif tasks:
                        msg.data = '好的，开始执行'
                    else:
                        msg.data = '没听清要夹哪个颜色或放到哪里'
                elif result and 'response' in result:
                    msg.data = result['response']
                else:
                    msg.data = self.llm_result
                self.tts_text_pub.publish(msg)
                self.action_finish = True
                self.llm_result = ''
            else:
                time.sleep(0.01)
            if self.play_audio_finish and self.action_finish:
                self.play_audio_finish = False
                self.action_finish = False
                msg = SetBool.Request()
                msg.data = True
                self.send_request(self.awake_client, msg)
        rclpy.shutdown()


def main():
    node = LLMColorSorting('llm_color_sorting')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()
