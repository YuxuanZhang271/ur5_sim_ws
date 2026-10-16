#!/usr/bin/env python3
"""
Publishes an initial joint trajectory at startup so the simulated robot
moves to a defined home pose before any user input.

Sequence
--------
1. (Optional) Wait for the first ``/joint_states`` message — confirms the
   simulation is up.
2. Publish a single-point ``JointTrajectory`` with the configured home
   positions on the controller's trajectory topic.
3. After a short grace period, shut rclpy down so the launch can proceed.

The Gazebo controller executes the trajectory, ``/joint_states`` follows,
and ``robot_state_publisher`` propagates the new transform into RViz —
so opening RViz after this node runs will display the home pose.
"""
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from endoscope_kinematics import JOINT_NAMES


class InitialPosePublisher(Node):

    def __init__(self):
        super().__init__('initial_pose_publisher')

        self.declare_parameter('controller_topic',
                               '/scaled_joint_trajectory_controller/'
                               'joint_trajectory')
        self.declare_parameter(
            'home_positions_deg', [0.0, -120.0, -120.0, -30.0, 90.0, 45.0])
        self.declare_parameter('trajectory_duration_s', 3.0)
        self.declare_parameter('wait_for_joint_states', True)
        self.declare_parameter('exit_delay_s', 1.0)

        self._home_rad = [
            math.radians(float(d))
            for d in self.get_parameter('home_positions_deg').value]
        self._duration_s = float(
            self.get_parameter('trajectory_duration_s').value)
        self._exit_delay_s = float(self.get_parameter('exit_delay_s').value)

        controller_topic = self.get_parameter('controller_topic').value
        self.traj_pub = self.create_publisher(JointTrajectory,
                                              controller_topic, 10)

        self.get_logger().info(
            f'Will publish home trajectory on {controller_topic}: '
            f'{self.get_parameter("home_positions_deg").value} deg, '
            f'duration {self._duration_s:.2f} s.')

        if self.get_parameter('wait_for_joint_states').value:
            self._joint_received = False
            self.create_subscription(
                JointState, '/joint_states', self._on_joint_state, 10)
            self.get_logger().info('Waiting for first /joint_states ...')
        else:
            self._publish_and_schedule_exit()

    def _on_joint_state(self, _msg: JointState):
        if self._joint_received:
            return
        self._joint_received = True
        self._publish_and_schedule_exit()

    def _publish_and_schedule_exit(self):
        self._publish_home()
        self.create_timer(self._exit_delay_s, self._shutdown)

    def _publish_home(self):
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in self._home_rad]
        pt.velocities = [0.0] * len(self._home_rad)
        pt.accelerations = [0.0] * len(self._home_rad)
        pt.time_from_start = Duration(
            sec=int(self._duration_s),
            nanosec=int((self._duration_s % 1.0) * 1e9))
        msg.points.append(pt)
        self.traj_pub.publish(msg)
        self.get_logger().info('Home trajectory published.')

    @staticmethod
    def _shutdown():
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    InitialPosePublisher()
    rclpy.spin()


if __name__ == '__main__':
    main()