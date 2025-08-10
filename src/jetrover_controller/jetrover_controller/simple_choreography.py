#!/usr/bin/env python3
"""
Simple Multi-Robot Choreography Controller

This controller executes a specific choreography:
- Both robots start on the same vertical line
- Robot 1: RIGHT (primary_time) → LEFT (secondary_time - primary_time)
- Robot 2: LEFT (primary_time) → RIGHT (secondary_time - primary_time)
- Final alignment to ensure both robots end on the same vertical line

Usage:
python3 simple_choreography.py --robots robot1,robot2
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import time
import math

class SimpleChoreographyController(Node):
    def __init__(self, robot_namespaces=None):
        super().__init__('simple_choreography_controller')
        
        # Default to robot1 and robot2 if no namespaces provided
        if robot_namespaces is None:
            robot_namespaces = ['robot_1', 'robot_2']
        
        self.robot_namespaces = robot_namespaces
        
        # Create publishers for each robot
        self.robot_publishers = {}
        for namespace in robot_namespaces:
            topic_name = f'/{namespace}/controller/cmd_vel'
            self.robot_publishers[namespace] = self.create_publisher(Twist, topic_name, 10)
            print(f"📡 Publisher created for {namespace}: {topic_name}")
        
        # Position tracking for each robot
        self.robot_positions = {}
        for namespace in robot_namespaces:
            self.robot_positions[namespace] = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        
        # Choreography parameters (hardcoded for simplicity)
        self.primary_time = 3.0    # 3 seconds
        self.secondary_time = 9.0  # 9 seconds total (so 6 more seconds)
        self.speed = 0.2          # 0.2 m/s
        
        print("🎭 Simple Multi-Robot Choreography Controller")
        print(f"🤖 Controlling Robots: {', '.join(robot_namespaces)}")
        print("="*60)
        print("📋 Choreography Details:")
        print(f"   Primary time: {self.primary_time}s")
        print(f"   Secondary time: {self.secondary_time}s (total)")
        print(f"   Speed: {self.speed} m/s")
        print("")
        print("🎯 Movement Pattern:")
        print("   Robot 1: RIGHT → LEFT")
        print("   Robot 2: LEFT → RIGHT")
        print("   Final: Auto-alignment to vertical line")
        print("="*60)
        print("Controls:")
        print("  E/e: Execute choreography")
        print("  S/s: Emergency STOP")
        print("  X/x: Exit")
        print("="*60)
        
    def update_position(self, robot_namespace, linear_x, linear_y, angular_z, duration):
        """Update robot position based on movement"""
        if robot_namespace not in self.robot_positions:
            return
            
        pos = self.robot_positions[robot_namespace]
        pos['x'] += linear_x * duration
        pos['y'] += linear_y * duration
        pos['theta'] += angular_z * duration
        pos['theta'] = math.atan2(math.sin(pos['theta']), math.cos(pos['theta']))
        
    def get_position(self, robot_namespace):
        """Get current robot position"""
        if robot_namespace not in self.robot_positions:
            return None
        return self.robot_positions[robot_namespace].copy()
        
    def reset_all_positions(self):
        """Reset all robot positions to origin"""
        for namespace in self.robot_positions:
            self.robot_positions[namespace] = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        print("🔄 All robot positions reset to origin")
        
    def send_cmd_to_robot(self, robot_namespace, linear_x=0, linear_y=0, angular_z=0):
        """Send movement command to specific robot"""
        if robot_namespace in self.robot_publishers:
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.linear.y = float(linear_y)
            twist.angular.z = float(angular_z)
            self.robot_publishers[robot_namespace].publish(twist)
    
    def send_cmd_to_all(self, linear_x=0, linear_y=0, angular_z=0):
        """Send movement command to all robots"""
        for namespace in self.robot_namespaces:
            self.send_cmd_to_robot(namespace, linear_x, linear_y, angular_z)
    
    def emergency_stop(self):
        """Emergency stop all robots"""
        print("🚨 EMERGENCY STOP - All robots stopped!")
        self.send_cmd_to_all(0, 0, 0)
        
    def execute_choreography(self):
        """Execute the specific multi-robot choreography"""
        print("🎯 Starting choreography execution...")
        print(f"⏱️  Primary: {self.primary_time}s, Secondary: {self.secondary_time}s, Speed: {self.speed}m/s")
        
        # Reset positions
        self.reset_all_positions()
        
        # Phase 1: Primary movement (3 seconds)
        print(f"\n🚀 Phase 1: Primary movement ({self.primary_time}s)")
        print("   Robot 1: Moving RIGHT")
        print("   Robot 2: Moving LEFT")
        
        start_time = time.time()
        while time.time() - start_time < self.primary_time:
            # Robot 1 moves RIGHT (negative Y)
            self.send_cmd_to_robot(self.robot_namespaces[0], linear_y=-self.speed)
            
            # Robot 2 moves LEFT (positive Y) 
            if len(self.robot_namespaces) > 1:
                self.send_cmd_to_robot(self.robot_namespaces[1], linear_y=self.speed)
            
            time.sleep(0.1)
        
        # Update positions after phase 1
        self.update_position(self.robot_namespaces[0], 0, -self.speed, 0, self.primary_time)
        if len(self.robot_namespaces) > 1:
            self.update_position(self.robot_namespaces[1], 0, self.speed, 0, self.primary_time)
        
        print("✅ Phase 1 complete")
        
        # Phase 2: Secondary movement (remaining time)
        remaining_time = self.secondary_time - self.primary_time
        print(f"\n🚀 Phase 2: Secondary movement ({remaining_time}s)")
        print("   Robot 1: Moving LEFT")
        print("   Robot 2: Moving RIGHT")
        
        start_time = time.time()
        while time.time() - start_time < remaining_time:
            # Robot 1 moves LEFT (positive Y)
            self.send_cmd_to_robot(self.robot_namespaces[0], linear_y=self.speed)
            
            # Robot 2 moves RIGHT (negative Y)
            if len(self.robot_namespaces) > 1:
                self.send_cmd_to_robot(self.robot_namespaces[1], linear_y=-self.speed)
            
            time.sleep(0.1)
        
        # Update positions after phase 2
        self.update_position(self.robot_namespaces[0], 0, self.speed, 0, remaining_time)
        if len(self.robot_namespaces) > 1:
            self.update_position(self.robot_namespaces[1], 0, -self.speed, 0, remaining_time)
        
        print("✅ Phase 2 complete")
        
        # Stop all robots
        self.send_cmd_to_all(0, 0, 0)
        
        # Show final positions
        print(f"\n📍 Final Positions:")
        for namespace in self.robot_namespaces:
            pos = self.get_position(namespace)
            if pos:
                print(f"   {namespace}: X={pos['x']:.3f}m, Y={pos['y']:.3f}m")
        
        # Phase 3: Alignment
        print(f"\n🎯 Phase 3: Auto-alignment to vertical line")
        
        for namespace in self.robot_namespaces:
            pos = self.get_position(namespace)
            if pos and abs(pos['y']) > 0.01:  # If not aligned (within 1cm)
                alignment_time = abs(pos['y']) / self.speed
                alignment_direction = -1 if pos['y'] > 0 else 1
                
                print(f"🔄 Aligning {namespace}: moving {'left' if alignment_direction > 0 else 'right'} for {alignment_time:.2f}s")
                
                start_time = time.time()
                while time.time() - start_time < alignment_time:
                    self.send_cmd_to_robot(namespace, linear_y=alignment_direction * self.speed)
                    time.sleep(0.1)
                
                # Stop this robot and update position
                self.send_cmd_to_robot(namespace, 0, 0, 0)
                self.update_position(namespace, 0, alignment_direction * self.speed, 0, alignment_time)
                
                final_pos = self.get_position(namespace)
                print(f"✅ {namespace} aligned. Final Y: {final_pos['y']:.3f}m")
            else:
                print(f"✅ {namespace} already aligned")
        
        # Final stop
        self.send_cmd_to_all(0, 0, 0)
        
        print(f"\n🎉 Choreography execution complete!")
        print("📍 Final aligned positions:")
        for namespace in self.robot_namespaces:
            pos = self.get_position(namespace)
            if pos:
                print(f"   {namespace}: X={pos['x']:.3f}m, Y={pos['y']:.3f}m")
    
    def get_key(self):
        """Get single keypress without Enter"""
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)
        return key
        
    def run(self):
        """Main control loop"""
        self.original_settings = termios.tcgetattr(sys.stdin)
        
        try:
            while True:
                print(f"\n⌨️  Waiting for command (E=Execute, S=Stop, X=Exit)...")
                key = self.get_key().lower()
                
                if key == 'e':
                    print("🚀 Executing choreography...")
                    self.execute_choreography()
                    
                elif key == 's':
                    self.emergency_stop()
                    
                elif key == 'x' or key == '\x03':  # x or Ctrl+C
                    print("👋 Exiting...")
                    break
                    
                else:
                    print(f"❓ Unknown key '{key}'. Use E=Execute, S=Stop, X=Exit")
                    
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
        finally:
            # Stop all robots and restore terminal
            self.send_cmd_to_all(0, 0, 0)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)

def main(args=None):
    # Parse ROS arguments
    rclpy.init(args=args)
    
    # Parse remaining arguments for robot namespaces
    robot_namespaces = ['robot1', 'robot2']  # Default
    
    # Simple argument parsing for --robots
    if '--robots' in sys.argv:
        try:
            robots_idx = sys.argv.index('--robots')
            if robots_idx + 1 < len(sys.argv):
                robot_namespaces = sys.argv[robots_idx + 1].split(',')
        except (ValueError, IndexError):
            pass
    
    try:
        controller = SimpleChoreographyController(robot_namespaces)
        controller.run()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
