#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from servo_controller_msgs.msg import ServosPosition
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import sys

class ArmSynchronizer(Node):
    def __init__(self, leader_robot='robot1', follower_robot='robot2'):
        super().__init__('arm_synchronizer')
        
        self.leader_robot = leader_robot
        self.follower_robot = follower_robot
        
        # Real-time QoS for minimal latency - compatible with existing subscribers
        realtime_qos = QoSProfile(
            depth=1,  # Keep only latest message
            reliability=QoSReliabilityPolicy.RELIABLE,  # Match subscriber expectations
            history=QoSHistoryPolicy.KEEP_LAST  # Replace old messages immediately
        )
        
        # Subscribe to leader robot's servo commands
        leader_topic = f'/{leader_robot}/servo_controller'
        self.subscriber_ = self.create_subscription(
            ServosPosition, 
            leader_topic,
            self.servo_command_callback, 
            realtime_qos
        )
        
        # Publish to follower robot's servo controller
        follower_topic = f'/{follower_robot}/servo_controller'
        self.publisher_ = self.create_publisher(
            ServosPosition, 
            follower_topic,
            realtime_qos
        )
        
        self.get_logger().info("🦾 Arm Synchronizer started!")
        self.get_logger().info(f"  👑 Leader: {leader_robot}")
        self.get_logger().info(f"  👥 Follower: {follower_robot}")
        self.get_logger().info(f"  → Listening to: {leader_topic}")
        self.get_logger().info(f"  → Publishing to: {follower_topic}")
        
    def servo_command_callback(self, msg):
        """Copy leader robot's servo commands to follower robot"""
        # Forward the exact same servo command to follower robot
        self.publisher_.publish(msg)
        
        # Log the synchronized movement
        if len(msg.position) > 0:
            servo_info = []
            for servo in msg.position:
                servo_info.append(f"ID{servo.id}:{servo.position:.0f}")
            
            self.get_logger().info(
                f"🔄 Syncing arm: [{', '.join(servo_info)}] "
                f"duration={msg.duration:.1f}s"
            )

def main():
    rclpy.init()
    
    # Parse command line arguments for leader and follower
    leader_robot = 'robot1'
    follower_robot = 'robot2'
    
    # Simple argument parsing for --leader and --follower
    if '--leader' in sys.argv:
        try:
            leader_idx = sys.argv.index('--leader')
            if leader_idx + 1 < len(sys.argv):
                leader_robot = sys.argv[leader_idx + 1]
        except (ValueError, IndexError):
            pass
    
    if '--follower' in sys.argv:
        try:
            follower_idx = sys.argv.index('--follower')
            if follower_idx + 1 < len(sys.argv):
                follower_robot = sys.argv[follower_idx + 1]
        except (ValueError, IndexError):
            pass
    
    synchronizer = ArmSynchronizer(leader_robot, follower_robot)
    
    try:
        rclpy.spin(synchronizer)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Arm Synchronizer...")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()