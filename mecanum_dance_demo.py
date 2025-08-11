#!/usr/bin/env python3
"""
Mecanum Dance Capabilities Demo
Shows all the unique movements available for the enhanced choreography system
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from servo_controller_msgs.msg import ServosPosition, ServoPosition
import time
import random

class MecanumDanceDemo(Node):
    def __init__(self):
        super().__init__('mecanum_dance_demo')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.servo_pub = self.create_publisher(ServosPosition, 'servo_controller', 10)
        
        self.get_logger().info("🎭 Mecanum Dance Capabilities Demo Ready!")
        
    def demo_movement(self, name, description, duration=3.0):
        """Demo a movement with description"""
        self.get_logger().info(f"🎪 {name}: {description}")
        time.sleep(1.0)  # Pause between movements
        
    def send_base_cmd(self, linear_x=0.0, linear_y=0.0, angular_z=0.0, duration=2.0):
        """Send base movement command"""
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        
        # Send for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        
        # Stop
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        
    def send_servo_cmd(self, positions, duration=2.0):
        """Send servo movement command"""
        msg = ServosPosition()
        msg.position_unit = 'pulse'
        msg.duration = duration
        
        for sid, pos in positions.items():
            servo_pos = ServoPosition()
            servo_pos.id = sid
            servo_pos.position = float(pos)
            msg.position.append(servo_pos)
            
        self.servo_pub.publish(msg)
        time.sleep(duration + 0.5)
    
    def demo_sideways_slide(self):
        """Demo: Sideways Slide - Pure lateral movement"""
        self.demo_movement(
            "Sideways Slide", 
            "Smooth lateral movement - unique to Mecanum wheels!"
        )
        
        # Left slide
        self.get_logger().info("  → Sliding LEFT")
        self.send_base_cmd(linear_y=0.15, duration=2.0)
        
        # Right slide  
        self.get_logger().info("  → Sliding RIGHT")
        self.send_base_cmd(linear_y=-0.15, duration=2.0)
        
    def demo_diagonal_drift(self):
        """Demo: Diagonal Drift - Forward + sideways simultaneously"""
        self.demo_movement(
            "Diagonal Drift",
            "Forward-diagonal movement - signature Mecanum capability!"
        )
        
        # Forward-left diagonal
        self.get_logger().info("  → Drifting FORWARD-LEFT")
        self.send_base_cmd(linear_x=0.12, linear_y=0.08, duration=2.0)
        
        # Forward-right diagonal
        self.get_logger().info("  → Drifting FORWARD-RIGHT") 
        self.send_base_cmd(linear_x=0.12, linear_y=-0.08, duration=2.0)
        
    def demo_spin_in_place(self):
        """Demo: Spin in Place - Pure rotation without translation"""
        self.demo_movement(
            "Spin in Place",
            "Pure rotation without moving position - perfect for dramatic moments!"
        )
        
        # Clockwise spin
        self.get_logger().info("  → Spinning CLOCKWISE")
        self.send_base_cmd(angular_z=-0.8, duration=2.5)
        
        # Counter-clockwise spin
        self.get_logger().info("  → Spinning COUNTER-CLOCKWISE")
        self.send_base_cmd(angular_z=0.8, duration=2.5)
        
    def demo_circular_flow(self):
        """Demo: Circular Flow - Move forward while rotating"""
        self.demo_movement(
            "Circular Flow",
            "Forward movement + rotation = beautiful circular path!"
        )
        
        self.get_logger().info("  → Flowing in CIRCULAR path")
        self.send_base_cmd(linear_x=0.1, angular_z=0.6, duration=4.0)
        
    def demo_zigzag_dance(self):
        """Demo: Zigzag Dance - Sharp directional changes"""
        self.demo_movement(
            "Zigzag Dance", 
            "Sharp directional changes - energetic and dynamic!"
        )
        
        directions = [
            ("LEFT", {'linear_y': 0.15}),
            ("RIGHT", {'linear_y': -0.15}),
            ("FORWARD", {'linear_x': 0.15}),
            ("BACKWARD", {'linear_x': -0.15})
        ]
        
        for direction_name, cmd in directions:
            self.get_logger().info(f"  → Burst {direction_name}")
            self.send_base_cmd(duration=0.8, **cmd)
            time.sleep(0.3)
            
    def demo_explosive_burst(self):
        """Demo: Explosive Burst - Quick directional movements"""
        self.demo_movement(
            "Explosive Burst",
            "Quick, sudden directional bursts - perfect for beat drops!"
        )
        
        # Random explosive movements
        for i in range(4):
            direction = random.choice([
                ("FORWARD BURST", {'linear_x': 0.25}),
                ("SIDE BURST LEFT", {'linear_y': 0.25}), 
                ("SIDE BURST RIGHT", {'linear_y': -0.25}),
                ("SPIN BURST", {'angular_z': 1.2})
            ])
            
            self.get_logger().info(f"  → {direction[0]}")
            self.send_base_cmd(duration=0.6, **direction[1])
            time.sleep(0.4)
            
    def demo_combined_movements(self):
        """Demo: Combined servo + base movements"""
        self.demo_movement(
            "Combined Dance",
            "Servo arms + Mecanum base working together!"
        )
        
        # Wave arms while sliding sideways
        self.get_logger().info("  → Arms waving + sliding left")
        servo_positions = {1: 400, 2: 600, 3: 350, 4: 650, 5: 450}
        self.send_servo_cmd(servo_positions, duration=1.5)
        self.send_base_cmd(linear_y=0.1, duration=2.0)
        
        # Dramatic arm sweep while spinning
        self.get_logger().info("  → Dramatic sweep + spinning")
        servo_positions = {1: 300, 2: 700, 3: 250, 4: 750, 5: 200}
        self.send_servo_cmd(servo_positions, duration=2.0)
        self.send_base_cmd(angular_z=0.5, duration=3.0)
        
        # Return to home
        home_positions = {1: 500, 2: 500, 3: 500, 4: 500, 5: 500}
        self.send_servo_cmd(home_positions, duration=1.5)
        
    def run_full_demo(self):
        """Run the complete Mecanum dance demo"""
        self.get_logger().info("🚀 Starting Mecanum Dance Capabilities Demo!")
        self.get_logger().info("=" * 60)
        
        # Basic Mecanum movements
        self.demo_sideways_slide()
        self.demo_diagonal_drift() 
        self.demo_spin_in_place()
        self.demo_circular_flow()
        
        # Advanced movements
        self.demo_zigzag_dance()
        self.demo_explosive_burst()
        
        # Combined movements
        self.demo_combined_movements()
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🎉 Demo Complete! These movements are now integrated into the AI choreography system!")
        self.get_logger().info("🎵 The system will automatically choose appropriate movements based on musical features:")
        self.get_logger().info("  • High energy + bright = Explosive bursts & spins")
        self.get_logger().info("  • Flowing melodies = Diagonal drifts & circular flows") 
        self.get_logger().info("  • Rhythmic sections = Zigzag dance & rhythmic steps")
        self.get_logger().info("  • Gentle parts = Smooth glides & subtle movements")

def main():
    rclpy.init()
    
    try:
        demo = MecanumDanceDemo()
        demo.run_full_demo()
        
    except KeyboardInterrupt:
        demo.get_logger().info("Demo interrupted by user")
    finally:
        # Stop everything
        stop_twist = Twist()
        demo.cmd_vel_pub.publish(stop_twist)
        rclpy.shutdown()

if __name__ == '__main__':
    main()
