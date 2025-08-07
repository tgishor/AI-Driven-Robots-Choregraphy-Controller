#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time

class RobotSynchronizer(Node):
    def __init__(self):
        super().__init__('robot_synchronizer')
        
        # Real-time QoS for minimal latency - compatible with existing subscribers
        realtime_qos = QoSProfile(
            depth=1,  # Keep only latest message
            reliability=QoSReliabilityPolicy.RELIABLE,  # Match subscriber expectations
            history=QoSHistoryPolicy.KEEP_LAST  # Replace old messages immediately
        )
        
        # Subscribe to Robot 1's commands (from network)
        self.subscriber_ = self.create_subscription(
            Twist, 
            '/robot1/controller/cmd_vel',  # Listen to Robot 1's topic
            self.cmd_vel_callback, 
            realtime_qos
        )
        
        # Publish to local robot's controller
        self.publisher_ = self.create_publisher(
            Twist, 
            '/robot2/controller/cmd_vel',  # Publish to local robot
            realtime_qos
        )
        
        # Command timing tracking
        # Note: last_cmd_time will be set on first command
        
        self.get_logger().info("🤖 Robot Follower started!")
        self.get_logger().info("  → Listening to:  /robot1/controller/cmd_vel")
        self.get_logger().info("  → Publishing to: /robot2/controller/cmd_vel (local)")
        self.get_logger().info("  ⚡ Real-time QoS enabled for minimal latency")
        
    def cmd_vel_callback(self, msg):
        """Copy Robot 1's commands to this robot"""
        receive_time = time.time()
        
        # Forward the exact same command to local robot (ASAP - minimize delay)
        self.publisher_.publish(msg)
        
        # Calculate time gap between commands (not latency)
        time_gap = receive_time - self.last_cmd_time if hasattr(self, 'last_cmd_time') else 0
        self.last_cmd_time = receive_time
        
        # Movement logging with timing info
        if msg.linear.x != 0 or msg.linear.y != 0 or msg.angular.z != 0:
            if time_gap < 1.0:  # Only log if reasonable gap (< 1 second)
                self.get_logger().info(
                    f"Following: {msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.angular.z:.2f} "
                    f"(gap: {time_gap*1000:.1f}ms)"
                )
            else:
                self.get_logger().info(
                    f"Following: {msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.angular.z:.2f} "
                    f"(first cmd after {time_gap:.1f}s pause)"
                )

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
