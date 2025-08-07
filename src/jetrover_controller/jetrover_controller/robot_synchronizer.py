#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time

class RobotSynchronizer(Node):
    def __init__(self):
        super().__init__('robot_synchronizer')
        
        # Real-time QoS for minimal latency
        realtime_qos = QoSProfile(
            depth=1,  # Keep only latest message
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Don't wait for acks
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
        
        # Latency tracking
        self.last_cmd_time = time.time()
        self.latency_samples = []
        self.max_samples = 100
        
        self.get_logger().info("🤖 Robot Follower started!")
        self.get_logger().info("  → Listening to:  /robot1/controller/cmd_vel")
        self.get_logger().info("  → Publishing to: /robot2/controller/cmd_vel (local)")
        self.get_logger().info("  ⚡ Real-time QoS enabled for minimal latency")
        
    def cmd_vel_callback(self, msg):
        """Copy Robot 1's commands to this robot"""
        current_time = time.time()
        
        # Calculate processing latency (time since last command)
        processing_latency = current_time - self.last_cmd_time
        self.last_cmd_time = current_time
        
        # Track latency samples
        if len(self.latency_samples) >= self.max_samples:
            self.latency_samples.pop(0)
        self.latency_samples.append(processing_latency)
        
        # Forward the exact same command to local robot (ASAP)
        self.publisher_.publish(msg)
        
        # Movement logging with latency info
        if msg.linear.x != 0 or msg.linear.y != 0 or msg.angular.z != 0:
            avg_latency = sum(self.latency_samples) / len(self.latency_samples) * 1000
            self.get_logger().info(
                f"Following: {msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.angular.z:.2f} "
                f"(latency: {processing_latency*1000:.1f}ms, avg: {avg_latency:.1f}ms)"
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
