#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from servo_controller_msgs.msg import ServosPosition, ServoPosition
from servo_controller_msgs.msg import ServoStateList
import time
import sys

class MultiRobotArmController(Node):

    def __init__(self, robot_namespace=''):
        # Create unique node name with namespace
        node_name = 'multi_robot_arm_controller'
        if robot_namespace:
            node_name = f'multi_robot_arm_controller_{robot_namespace}'
            
        super().__init__(node_name)
        
        # Build topic names with namespace
        if robot_namespace:
            servo_controller_topic = f'/{robot_namespace}/servo_controller'
            servo_states_topic = f'/{robot_namespace}/controller_manager/servo_states'
        else:
            servo_controller_topic = 'servo_controller'
            servo_states_topic = '/servo_states'
            
        self.publisher_ = self.create_publisher(ServosPosition, servo_controller_topic, 10)
        self.robot_namespace = robot_namespace

        self.current_pose = {}
        self.pose_received = False

        self.create_subscription(
            ServoStateList,
            servo_states_topic,
            self.servo_state_callback,
            10
        )
        
        print("🦾 Multi-Robot Arm Controller")
        if robot_namespace:
            print(f"🏷️  Controlling Robot: {robot_namespace}")
        print(f"📤 Publishing to: {servo_controller_topic}")
        print(f"📥 Listening to: {servo_states_topic}")

    def servo_state_callback(self, msg):
        for servo in msg.servo_state:
            self.current_pose[servo.id] = servo.position
        self.pose_received = True
    
    def is_default_pose(self, default_pose: dict, tolerance: int = 10):
        """
        Check if the current servo pose is close enough to the default.
        :param default_pose: Dict of {servo_id: target_position}
        :param tolerance: Allowed pulse error (e.g. ±10)
        :return: True if all match, False otherwise
        """
        for servo_id, target in default_pose.items():
            current = self.current_pose.get(servo_id, None)
            if current is None:
                self.get_logger().warn(f"No reading for servo ID {servo_id}")
                return False
            if abs(current - target) > tolerance:
                return False
        return True

    def move_arm(self, duration_ms=1000, **servo_positions):
        """
        Send pulse commands to multiple servos with optional duration.
        Usage:
            move_arm(duration_ms=1000, joint1=500, joint2=600, gripper=700)
        Servo IDs must match your hardware mapping.
        """
        msg = ServosPosition()
        msg.position_unit = 'pulse'
        msg.duration = duration_ms/1000.0

        # Mapping from friendly names to actual servo IDs
        servo_map = {
            'joint1': 1,
            'joint2': 2,
            'joint3': 3,
            'joint4': 4,
            'joint5': 5,
            'gripper': 10,
        }

        for name, pulse in servo_positions.items():
            if name in servo_map:
                servo = ServoPosition()
                servo.id = servo_map[name]
                servo.position = float(pulse)
                msg.position.append(servo)
            else:
                self.get_logger().warn(f"Unknown joint name: {name}")

        self.publisher_.publish(msg)
        self.get_logger().info(f"Sent command: {servo_positions} with duration {duration_ms} ms")

    def default_move(self):
        # Just one example: move to a certain pose
        self.move_arm(
            duration_ms=1500,
            joint1=500,
            joint2=500,
            joint3=500,
            joint4=500,
            joint5=500,
            gripper=750
        )

    def demo_sequence(self):
        """Run a demo sequence of arm movements"""
        self.get_logger().info("🎬 Starting demo sequence...")
        
        # Position 1: Default/Home position
        self.get_logger().info("📍 Moving to home position...")
        self.move_arm(
            duration_ms=2000,
            joint1=500,
            joint2=500,
            joint3=500,
            joint4=500,
            joint5=500,
            gripper=750
        )
        time.sleep(3)
        
        # Position 2: Extended position
        self.get_logger().info("📍 Moving to extended position...")
        self.move_arm(
            duration_ms=2000,
            joint1=300,
            joint2=200,
            joint3=200,
            joint4=200,
            joint5=300,
            gripper=400
        )
        time.sleep(3)
        
        # Position 3: Reach up
        self.get_logger().info("📍 Reaching up...")
        self.move_arm(
            duration_ms=2000,
            joint1=500,
            joint2=700,
            joint3=700,
            joint4=700,
            joint5=600,
            gripper=600
        )
        time.sleep(3)
        
        # Position 4: Back to home
        self.get_logger().info("📍 Returning to home...")
        self.move_arm(
            duration_ms=2000,
            joint1=500,
            joint2=500,
            joint3=500,
            joint4=500,
            joint5=500,
            gripper=750
        )
        
        self.get_logger().info("✅ Demo sequence completed!")

def main(args=None):
    rclpy.init(args=args)
    
    # Parse remaining arguments for robot namespace
    robot_namespace = ''
    
    # Simple argument parsing for --robot
    if '--robot' in sys.argv:
        try:
            robot_idx = sys.argv.index('--robot')
            if robot_idx + 1 < len(sys.argv):
                robot_namespace = sys.argv[robot_idx + 1]
        except (ValueError, IndexError):
            pass
    
    node = MultiRobotArmController(robot_namespace)
    time.sleep(1.0)
    
    try:
        # Run demo sequence once
        node.demo_sequence()
        
        # Keep node alive for manual control if needed
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping Multi-Robot Arm Controller...")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()