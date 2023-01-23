import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
from dynamic_reconfigure.server import Server
# from tracking_pid.cfg import ParamsConfig
from waypts_nav_pid.cfg import ParamsConfig 
# from tracking_param.cfg import ParamsConfig


class TrackingPID:
    def __init__(self):
        # Node initialization
        rospy.init_node("tracking_pid")
        # Subscribers and Publishers
        self.sub_odom = rospy.Subscriber(
            # "/odometry/filtered",
            "/ground_truth/state",
            Odometry,
            self.odom_callback,
        )
        self.sub_waypoint = rospy.Subscriber(
            "/waypoint",
            PointStamped,
            self.waypoint_callback,
        )
        self.sub_rviz_goal = rospy.Subscriber(
            "/move_base_simple/goal",
            PoseStamped,
            self.rviz_goal_callback,
        )
        self.pub_waypoint = rospy.Publisher(
            "/waypoint_marker",
            Marker,
            queue_size=1,
        )
        self.pub_flag = rospy.Publisher(
            "/need_waypoint",
            Bool,
            queue_size=1,
        )
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        # Data
        self.twist = Twist()
        self.need_waypoint = Bool()
        self.need_waypoint.data = True
        self.init_waypoint_marker()
        self.odom = np.empty(3)
        self.waypoint = self.odom[:2].copy()
        self.trim_lin_vel = 0.0
        self.verbose = True
        # Others
        self.angular_tolerance = 0.5
        self.robot_radius = 0.5 ## jackal .5 # heron 2 m 
        self.pid_angular = PID(
            kp=0.5,
            kd=0.0,
            ki=0.0,
            min_output=-0.35,
            max_output=0.35,
            delta=0.01,
            min_integral=-0.0,
            max_integral=0.0,
            name="Angular",
        )
        self.pid_linear = PID(
            kp=0.7,
            kd=0.0,
            ki=0.0,
            min_output=-1.0,
            max_output=1.0,
            delta=0.04,
            min_integral=-0.0,
            max_integral=0.0,
            name="Linear",
        )
        # Spinning up
        Server(ParamsConfig, self.param_callback)
        rospy.on_shutdown(self.stop)
        rospy.spin()

    def rviz_goal_callback(self, msg):
        self.need_waypoint.data = False
        self.waypoint[0] = msg.pose.position.x
        self.waypoint[1] = msg.pose.position.y
        rospy.loginfo(msg)

    def param_callback(self, config, _):
        self.angular_tolerance = config["angular_tolerance"]
        self.robot_radius = config["robot_radius"]
        self.trim_lin_vel = config["trim_lin_vel"]
        self.verbose = config["verbose"]

        self.pid_angular.kp = config["angular_kp"]
        self.pid_angular.ki = config["angular_ki"]
        self.pid_angular.kd = config["angular_kd"]
        self.pid_angular.min_integral = config["angular_min_integral"]
        self.pid_angular.max_integral = config["angular_max_integral"]
        self.pid_angular.min_output = config["angular_min_output"]
        self.pid_angular.max_output = config["angular_max_output"]
        self.pid_angular.delta = config["angular_delta"]

        self.pid_linear.kp = config["linear_kp"]
        self.pid_linear.ki = config["linear_ki"]
        self.pid_linear.kd = config["linear_kd"]
        self.pid_linear.min_integral = config["linear_min_integral"]
        self.pid_linear.max_integral = config["linear_max_integral"]
        self.pid_linear.min_output = config["linear_min_output"]
        self.pid_linear.max_output = config["linear_max_output"]
        self.pid_linear.delta = config["linear_delta"]
        return config

    def init_waypoint_marker(self):
        self.waypoint_marker = Marker()
        self.waypoint_marker.header.frame_id = "odom"
        self.waypoint_marker.header.stamp = rospy.Time.now()
        self.waypoint_marker.type = self.waypoint_marker.SPHERE
        self.waypoint_marker.action = 0  # add or modify
        self.waypoint_marker.pose.orientation.x = 0.0
        self.waypoint_marker.pose.orientation.y = 0.0
        self.waypoint_marker.pose.orientation.z = 0.0
        self.waypoint_marker.pose.orientation.w = 1.0
        self.waypoint_marker.scale.x = 1.0
        self.waypoint_marker.scale.y = 1.0
        self.waypoint_marker.scale.z = 1.0
        self.waypoint_marker.color.r = 1.0
        self.waypoint_marker.color.g = 0.0
        self.waypoint_marker.color.b = 0.0
        self.waypoint_marker.color.a = 1.0

    def stop(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.pub_cmd_vel.publish(self.twist)

    def odom_callback(self, msg):
        self.odom[0] = msg.pose.pose.position.x
        self.odom[1] = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        self.odom[2] = euler_from_quaternion(quaternion)[2]
        if self.need_waypoint.data:
            rospy.loginfo("Waiting for new waypoint...")
        else:
            self.control()
        self.pub_flag.publish(self.need_waypoint)

    def waypoint_callback(self, msg):
        self.waypoint[0] = msg.point.x
        self.waypoint[1] = msg.point.y
        self.need_waypoint.data = False

    def control(self):
        self.waypoint_marker.pose.position.x = self.waypoint[0]
        self.waypoint_marker.pose.position.y = self.waypoint[1]
        self.pub_waypoint.publish(self.waypoint_marker)
        x_diff = self.waypoint[0] - self.odom[0]
        y_diff = self.waypoint[1] - self.odom[1]
        dist = np.hypot(y_diff, x_diff)

        print("\n\n distance error: ", dist)


        yaw = self.odom[2]
        x_relative = np.cos(yaw) * x_diff + np.sin(yaw) * y_diff
        y_relative = -np.sin(yaw) * x_diff + np.cos(yaw) * y_diff
        angular_error = np.arctan2(y_relative, x_relative)
        linear_error = np.tanh(x_relative)
        self.twist.angular.z = self.pid_angular.update(
            angular_error,
            self.verbose,
        )
        self.twist.linear.x = self.pid_linear.update(
            linear_error,
            self.verbose,
        )
        if np.abs(angular_error) > self.angular_tolerance:
            self.twist.linear.x = self.trim_lin_vel
        if dist < self.robot_radius:
            self.need_waypoint.data = True
            self.stop()
        self.pub_cmd_vel.publish(self.twist)


class PID:
    def __init__(
        self,
        kp,
        kd,
        ki,
        min_output,
        max_output,
        delta,
        min_integral,
        max_integral,
        name,
    ):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.min_output = min_output
        self.max_output = max_output
        self.delta = delta
        self.min_integral = min_integral
        self.max_integral = max_integral
        self.name = name
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_output = 0.0
        self.previous_error = None
        self.previous_time = rospy.Time.now()

    def update(self, error, verbose=True):
        current_time = rospy.Time.now()
        dt = current_time.to_sec() - self.previous_time.to_sec()
        proportional = self.kp * error
        derivative = 0.0
        if dt > 0:
            if self.previous_error is None:
                self.previous_error = error
            derivative = self.kd * (error - self.previous_error) / dt
        self.integral += error * dt
        self.integral = np.clip(
            self.integral,
            self.max_integral,
            self.max_integral,
        )
        integral = self.ki * self.integral
        output = np.clip(
            proportional + integral + derivative,
            self.min_output,
            self.max_output,
        )
        output = np.clip(
            output,
            self.previous_output - self.delta,
            self.previous_output + self.delta,
        )
        self.previous_error = error
        self.previous_time = current_time
        self.previous_output = output

        if verbose:
            debug_msg = f"{self.name} PID "
            debug_msg += f"error: {error: .2f} "
            debug_msg += f"p: {proportional: .2f} "
            debug_msg += f"i: {integral: .2f} "
            debug_msg += f"d: {derivative: .2f} "
            debug_msg += f"output: {output: .2f}"
            rospy.loginfo(debug_msg)
        return output


if __name__ == "__main__":
    try:
        TrackingPID()
    except rospy.ROSInterruptException:
        pass
