#!/usr/bin/env python3
"""
Emergency Stop Test Script
Run this to test if emergency stop is working
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import time

class EmergencyStopTester(Node):
    def __init__(self):
        super().__init__('emergency_stop_tester')
        self.stop_pub = self.create_publisher(Bool, '/dance/stop_command', 10)
        self.get_logger().info("Emergency Stop Tester ready!")
    
    def send_stop(self):
        """Send emergency stop command"""
        self.get_logger().warn("🚨 SENDING EMERGENCY STOP COMMAND!")
        
        stop_msg = Bool()
        stop_msg.data = True
        
        # Send multiple times to ensure it's received
        for i in range(10):
            self.stop_pub.publish(stop_msg)
            self.get_logger().warn(f"Stop command sent {i+1}/10")
            time.sleep(0.1)
        
        self.get_logger().warn("✅ Emergency stop commands sent!")

def main():
    rclpy.init()
    tester = EmergencyStopTester()
    
    print("\n🚨 EMERGENCY STOP TESTER 🚨")
    print("This will send emergency stop commands to the dance node")
    print("Press ENTER to send emergency stop...")
    input()
    
    tester.send_stop()
    
    # Keep spinning for a moment
    for i in range(10):
        rclpy.spin_once(tester, timeout_sec=0.1)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
