#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotSynchronizer(Node):
    def __init__(self):
        super().__init__('robot_synchronizer')
        
        # Subscribe to Robot 1's commands (from network)
        self.subscriber_ = self.create_subscription(
            Twist, 
            '/robot1/controller/cmd_vel',  # Listen to Robot 1's topic
            self.cmd_vel_callback, 
            10
        )
        
        # Publish to local robot's controller
        self.publisher_ = self.create_publisher(
            Twist, 
            '/robot2/controller/cmd_vel',  # Publish to local robot
            10
        )
        
        self.get_logger().info("🤖 Robot Follower started!")
        self.get_logger().info("  → Listening to:  /robot1/controller/cmd_vel")
        self.get_logger().info("  → Publishing to: /robot2/controller/cmd_vel (local)")
        
    def cmd_vel_callback(self, msg):
        """Copy Robot 1's commands to this robot"""
        # Forward the exact same command to local robot
        self.publisher_.publish(msg)
        
        # Simple movement log
        if msg.linear.x != 0 or msg.linear.y != 0 or msg.angular.z != 0:
            self.get_logger().info(f"Following: {msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.angular.z:.2f}")

def main():
    rclpy.init()
    synchronizer = RobotSynchronizer()
    
    try:
        rclpy.spin(synchronizer)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Robot Synchronizer...")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
