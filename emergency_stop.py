#!/usr/bin/env python3
"""
Emergency Stop Script for Advanced Dance Controller
Usage: python emergency_stop.py
"""
import rclpy
from std_msgs.msg import Bool

def main():
    rclpy.init()
    
    try:
        # Create a simple node to send emergency stop
        node = rclpy.create_node('emergency_stop_sender')
        
        # Create publisher for emergency stop
        stop_pub = node.create_publisher(Bool, '/dance/stop_command', 10)
        
        print("🚨 SENDING EMERGENCY STOP COMMAND 🚨")
        
        # Send stop command
        stop_msg = Bool()
        stop_msg.data = True
        stop_pub.publish(stop_msg)
        
        print("Emergency stop command sent!")
        print("All robot movements should halt immediately.")
        
    except Exception as e:
        print(f"Error sending emergency stop: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
