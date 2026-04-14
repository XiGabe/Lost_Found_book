#!/usr/bin/env python3
import threading
import time

import rclpy
from nav2_msgs.action import FollowWaypoints
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class WaypointLoopProxy(Node):
    def __init__(self):
        super().__init__('waypoint_loop_proxy')

        callback_group = ReentrantCallbackGroup()
        self._active_goal_lock = threading.Lock()
        self._has_active_goal = False
        self._current_goal_handle = None
        self._goal_finished_event = threading.Event()
        self._goal_finished_event.set()
        self.navigator = BasicNavigator()
        self.server = ActionServer(
            self,
            FollowWaypoints,
            'follow_waypoints',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=callback_group,
        )

        self.navigator.waitUntilNav2Active()
        self.get_logger().info('looping waypoint follower started on follow_waypoints')

    def goal_callback(self, goal_request):
        if not goal_request.poses:
            self.get_logger().warn('reject empty waypoint goal')
            return GoalResponse.REJECT

        with self._active_goal_lock:
            has_active_goal = self._has_active_goal
            current_goal_handle = self._current_goal_handle

        if has_active_goal:
            if current_goal_handle is not None and current_goal_handle.is_cancel_requested:
                self.get_logger().info('waiting for canceled waypoint goal to fully release')
                self._goal_finished_event.wait(timeout=2.0)
            with self._active_goal_lock:
                if self._has_active_goal:
                    self.get_logger().warn('reject new waypoint goal while loop is active')
                    return GoalResponse.REJECT
                self._has_active_goal = True
                self._goal_finished_event.clear()
            return GoalResponse.ACCEPT

        with self._active_goal_lock:
            self._has_active_goal = True
            self._goal_finished_event.clear()
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        self.get_logger().info('cancel waypoint loop requested')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        try:
            self._current_goal_handle = goal_handle
            loop_index = 0
            poses = list(goal_handle.request.poses)
            result = self._empty_result()
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self.navigator.cancelTask()
                    goal_handle.canceled()
                    return result

                loop_index += 1
                self.get_logger().info(f'starting waypoint loop #{loop_index}')

                missed_waypoints = []
                for index, pose in enumerate(poses):
                    feedback = FollowWaypoints.Feedback()
                    feedback.current_waypoint = index
                    goal_handle.publish_feedback(feedback)
                    self.get_logger().info(
                        f'loop #{loop_index}: navigating to waypoint {index + 1}/{len(poses)}'
                    )

                    self.navigator.goToPose(pose)
                    while not self.navigator.isTaskComplete():
                        if goal_handle.is_cancel_requested:
                            self.get_logger().info('cancel requested, stopping current waypoint task')
                            self.navigator.cancelTask()
                            time.sleep(0.2)
                            goal_handle.canceled()
                            result.missed_waypoints = missed_waypoints
                            return result

                        nav_feedback = self.navigator.getFeedback()
                        if nav_feedback and Duration.from_msg(nav_feedback.navigation_time) > Duration(seconds=600.0):
                            self.get_logger().warn('waypoint navigation timed out, canceling current goal')
                            self.navigator.cancelTask()
                        time.sleep(0.05)

                    nav_result = self.navigator.getResult()
                    if nav_result == TaskResult.SUCCEEDED:
                        continue

                    missed_waypoints.append(index)
                    self.get_logger().warn(
                        f'waypoint {index + 1} ended with status {nav_result}, continuing'
                    )

                result.missed_waypoints = missed_waypoints
                self.get_logger().info(
                    f'waypoint loop #{loop_index} finished, restarting from waypoint 1'
                )

            goal_handle.abort()
            return result
        finally:
            with self._active_goal_lock:
                self._has_active_goal = False
                self._current_goal_handle = None
            self._goal_finished_event.set()

    def _empty_result(self):
        result = FollowWaypoints.Result()
        if hasattr(result, 'missed_waypoints'):
            result.missed_waypoints = []
        return result


def main():
    rclpy.init()
    node = WaypointLoopProxy()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

