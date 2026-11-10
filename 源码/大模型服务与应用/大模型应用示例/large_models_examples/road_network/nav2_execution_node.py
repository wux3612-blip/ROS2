#!/usr/bin/env python3
# encoding: utf-8

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Bool
from nav_msgs.msg import Path
from nav2_msgs.action import FollowPath
from action_msgs.msg import GoalStatus

class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')

        self.get_logger().info('\033[1;32m>>> Navigation Controller Started (Waiting for /final_path_commands) <<<\033[0m')

        # --- Configuration (--- 配置 ---)
        self.controller_id = 'FollowPath' 
        self.goal_checker_id = 'goal_checker'
        
        # --- Communication Interfaces (--- 通信接口 ---)
        self._action_client = ActionClient(self, FollowPath, 'follow_path')
        self.create_subscription(Path, '/final_path_commands', self.path_callback, 1)
        self.reach_pub = self.create_publisher(Bool, '/navigation_controller/reach_goal', 1)
        self.path_pub = self.create_publisher(Path, '/generated_path', 1)

        self.goal_handle = None
        self.command_seq = 0

    def path_callback(self, msg):
        """Process path commands from upper layer (处理上层发送的路径指令) """
        path_length = len(msg.poses)
        
        if path_length == 0:
            self.get_logger().warn("Empty path received. Triggering emergency stop.")
            self.command_seq += 1
            self.cancel_current_goal() # Call cancel function (调用取消函数)
            return

        self.get_logger().info(f"New path received ({path_length} points). Starting execution...")
        self.command_seq += 1
        self.path_pub.publish(msg) # Forward path for Rviz visualization (转发路径供 Rviz 可视化) 
        self.cancel_current_goal()
        self.send_follow_path_goal(msg, self.command_seq)
    
    def cancel_current_goal(self):
        """Actively cancel current Nav2 task (主动取消当前正在执行的 Nav2 任务) """
        if self.goal_handle is not None and self.goal_handle.accepted:
            goal_handle = self.goal_handle
            future = goal_handle.cancel_goal_async()
            future.add_done_callback(lambda future: self.cancel_done_callback(future, goal_handle))
        

    def cancel_done_callback(self, future, goal_handle):
        if len(future.result().goals_canceling) > 0:
            self.get_logger().info('Goal successfully cancelled.')
            if self.goal_handle is goal_handle:
                self.goal_handle = None 

    def send_follow_path_goal(self, path_msg, command_seq):
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("FollowPath Action Server not connected!")
            if command_seq == self.command_seq:
                self.notify_reach_status(False)
            return

        goal_msg = FollowPath.Goal()
        goal_msg.path = path_msg
        goal_msg.controller_id = self.controller_id
        goal_msg.goal_checker_id = self.goal_checker_id

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(lambda future: self.goal_response_callback(future, command_seq))

    def goal_response_callback(self, future, command_seq):
        goal_handle = future.result()
        if command_seq != self.command_seq:
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            return

        if not goal_handle.accepted:
            self.get_logger().error('Path rejected by Controller!')
            self.goal_handle = None
            msg = Bool(); msg.data = False
            self.reach_pub.publish(msg)
            return

        self.goal_handle = goal_handle
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(lambda future: self.get_result_callback(future, command_seq))

    def feedback_callback(self, feedback_msg):
        # feedback = feedback_msg.feedback
        # self.get_logger().info(f'Distance to goal: {feedback.distance_to_goal:.2f} m', throttle_duration_sec=2.0) (self.get_logger().info(f'距离终点: {feedback.distance_to_goal:.2f}m', throttle_duration_sec=2.0))
        pass

    def get_result_callback(self, future, command_seq):
        """Task completed (success or failure) (任务结束（成功或失败）)"""
        if command_seq != self.command_seq:
            return

        status = future.result().status     
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('\033[1;32mPath segment completed successfully!\033[0m')
            self.notify_reach_status(True)  
        else:
            self.get_logger().warn(f'Path following failed with status code: {status}')
            self.notify_reach_status(False)
        self.goal_handle = None

    def notify_reach_status(self, success):
        """Sender Unified feedback of execution status to Sender (统一反馈执行状态给) """
        msg = Bool()
        msg.data = success
        self.reach_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = NavigationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
