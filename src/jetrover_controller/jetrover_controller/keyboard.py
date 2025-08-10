#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select
import argparse
import time
import threading
import math
import json
import os
from datetime import datetime

class MultiRobotTeleop(Node):
    def __init__(self, robot_namespace=''):
        # Create unique node name with namespace
        node_name = 'multi_robot_teleop'
        if robot_namespace:
            node_name = f'multi_robot_teleop_{robot_namespace}'
            
        super().__init__(node_name)
        
        # Build topic name with namespace
        if robot_namespace:
            topic_name = f'/{robot_namespace}/controller/cmd_vel'
        else:
            topic_name = 'controller/cmd_vel'
            
        self.publisher_ = self.create_publisher(Twist, topic_name, 10)
        self.robot_namespace = robot_namespace
        
        self.speed = 0.1
        self.turn_speed = 0.5  # Start conservative, use ] to increase
        
        # Recording functionality
        self.recording_mode = False
        self.current_move_sequence = []
        self.recorded_moves = {}
        self.moves_file = f"recorded_moves_{robot_namespace}.json" if robot_namespace else "recorded_moves.json"
        self.load_recorded_moves()
        
        print("🤖 Multi-Robot Keyboard Teleop")
        if robot_namespace:
            print(f"🏷️  Controlling Robot: {robot_namespace}")
        print("="*40)
        print("Controls:")
        print("  w/s: Forward/Backward")
        print("  a/d: Left/Right (Mecanum)")
        print("  q/e: Rotate Left/Right")
        print("  z/c: Diagonal movements")
        print("  +/-: Increase/Decrease speed")
        print("  [/]: Decrease/Increase turn speed")
        print("  r: Full 360° turn right (test)")
        print("  SPACE: Stop")
        print("  x: Exit")
        print("")
        print("🎬 Recording Controls:")
        print("  R: Start/Stop recording mode")
        print("  p: Play recorded move")
        print("  l: List recorded moves")
        print("  d: Delete recorded move")
        print("="*40)
        turn_time = (2 * math.pi) / self.turn_speed
        print(f"Current speed: {self.speed:.2f} m/s")
        print(f"Current turn speed: {self.turn_speed:.2f} rad/s (360° in {turn_time:.1f}s)")
        print(f"Publishing to: {topic_name}")
        
    def load_recorded_moves(self):
        """Load recorded moves from JSON file"""
        try:
            if os.path.exists(self.moves_file):
                with open(self.moves_file, 'r') as f:
                    self.recorded_moves = json.load(f)
                print(f"📂 Loaded {len(self.recorded_moves)} recorded moves from {self.moves_file}")
        except Exception as e:
            print(f"⚠️  Could not load recorded moves: {e}")
            self.recorded_moves = {}
            
    def save_recorded_moves(self):
        """Save recorded moves to JSON file"""
        try:
            with open(self.moves_file, 'w') as f:
                json.dump(self.recorded_moves, f, indent=2)
            print(f"💾 Saved {len(self.recorded_moves)} moves to {self.moves_file}")
        except Exception as e:
            print(f"❌ Error saving moves: {e}")
            
    def start_recording(self):
        """Start recording mode"""
        if not self.recording_mode:
            self.recording_mode = True
            self.current_move_sequence = []
            print("🎬 Recording mode STARTED")
            print("   Make your moves, then press SPACE to stop and name the sequence")
        else:
            print("⚠️  Already in recording mode")
            
    def stop_recording(self):
        """Stop recording mode and save the move"""
        if self.recording_mode:
            self.recording_mode = False
            if self.current_move_sequence:
                print(f"📝 Recorded {len(self.current_move_sequence)} moves")
                move_name = input("Enter move name: ").strip()
                if move_name:
                    self.recorded_moves[move_name] = {
                        'sequence': self.current_move_sequence,
                        'timestamp': datetime.now().isoformat(),
                        'speed': self.speed,
                        'turn_speed': self.turn_speed
                    }
                    self.save_recorded_moves()
                    print(f"✅ Saved move: {move_name}")
                else:
                    print("❌ No name provided, move discarded")
            else:
                print("⚠️  No moves recorded")
            self.current_move_sequence = []
        else:
            print("⚠️  Not in recording mode")
            
    def record_move(self, move_type, **kwargs):
        """Record a move with timestamp"""
        if self.recording_mode:
            move_data = {
                'type': move_type,
                'timestamp': time.time(),
                'speed': self.speed,
                'turn_speed': self.turn_speed,
                **kwargs
            }
            self.current_move_sequence.append(move_data)
            
    def play_move(self, move_name):
        """Play a recorded move sequence"""
        if move_name not in self.recorded_moves:
            print(f"❌ Move '{move_name}' not found")
            return
            
        move_data = self.recorded_moves[move_name]
        sequence = move_data['sequence']
        
        print(f"🎭 Playing move: {move_name} ({len(sequence)} actions)")
        
        # Play the sequence
        for i, action in enumerate(sequence):
            print(f"  [{i+1}/{len(sequence)}] {action['type']}")
            
            if action['type'] == 'forward':
                self.send_cmd(linear_x=action['speed'])
            elif action['type'] == 'backward':
                self.send_cmd(linear_x=-action['speed'])
            elif action['type'] == 'left':
                self.send_cmd(linear_y=action['speed'])
            elif action['type'] == 'right':
                self.send_cmd(linear_y=-action['speed'])
            elif action['type'] == 'rotate_left':
                self.send_cmd(angular_z=action['turn_speed'])
            elif action['type'] == 'rotate_right':
                self.send_cmd(angular_z=-action['turn_speed'])
            elif action['type'] == 'diagonal_fl':
                self.send_cmd(linear_x=action['speed'], linear_y=action['speed'])
            elif action['type'] == 'diagonal_fr':
                self.send_cmd(linear_x=action['speed'], linear_y=-action['speed'])
            elif action['type'] == 'stop':
                self.send_cmd()
                
            # Wait for the duration (if specified) or a short delay
            duration = action.get('duration', 0.1)
            time.sleep(duration)
            
        # Stop at the end
        self.send_cmd()
        print(f"✅ Finished playing: {move_name}")
        
    def list_moves(self):
        """List all recorded moves"""
        if not self.recorded_moves:
            print("📋 No recorded moves found")
            return
            
        print("📋 Recorded Moves:")
        print("-" * 50)
        for name, data in self.recorded_moves.items():
            sequence_length = len(data['sequence'])
            timestamp = data['timestamp']
            print(f"  {name}: {sequence_length} actions ({timestamp})")
        print("-" * 50)
        
    def delete_move(self, move_name):
        """Delete a recorded move"""
        if move_name in self.recorded_moves:
            del self.recorded_moves[move_name]
            self.save_recorded_moves()
            print(f"🗑️  Deleted move: {move_name}")
        else:
            print(f"❌ Move '{move_name}' not found")
        
    def get_key(self):
        """Get single keypress without Enter"""
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)
        return key
        
    def send_cmd(self, linear_x=0, linear_y=0, angular_z=0):
        """Send movement command"""
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.linear.y = float(linear_y)
        twist.angular.z = float(angular_z)
        self.publisher_.publish(twist)
        
    def full_turn_right(self):
        """Execute a full 360° turn to the right"""
        # Calculate time for 360° turn: 2π / angular_velocity
        turn_time = (2 * math.pi) / self.turn_speed
        
        print(f"🔄 Starting 360° turn right at {self.turn_speed:.2f} rad/s")
        print(f"⏱️  Expected time: {turn_time:.1f} seconds")
        
        # Start timing
        start_time = time.time()
        
        # Run in separate thread to not block keyboard input
        def execute_turn():
            # Send rotation commands at 10Hz for the calculated duration
            rate = 10  # Hz
            total_iterations = int(turn_time * rate)
            
            for i in range(total_iterations):
                self.send_cmd(angular_z=-self.turn_speed)  # Negative for right turn
                time.sleep(1.0 / rate)
            
            # Stop the robot
            self.send_cmd()  # Send stop command
            
            actual_time = time.time() - start_time
            print(f"✅ Turn completed! Actual time: {actual_time:.1f}s (Expected: {turn_time:.1f}s)")
        
        # Execute in background thread
        turn_thread = threading.Thread(target=execute_turn, daemon=True)
        turn_thread.start()
        
    def run(self):
        """Main control loop"""
        self.original_settings = termios.tcgetattr(sys.stdin)
        
        try:
            while True:
                key = self.get_key()
                
                if key == 'w':
                    print(f"⬆️  Forward ({self.speed:.2f})")
                    self.send_cmd(linear_x=self.speed)
                    self.record_move('forward', speed=self.speed)
                    
                elif key == 's':
                    print(f"⬇️  Backward ({self.speed:.2f})")
                    self.send_cmd(linear_x=-self.speed)
                    self.record_move('backward', speed=self.speed)
                    
                elif key == 'a':
                    print(f"⬅️  Left ({self.speed:.2f})")
                    self.send_cmd(linear_y=self.speed)
                    self.record_move('left', speed=self.speed)
                    
                elif key == 'd':
                    print(f"➡️  Right ({self.speed:.2f})")
                    self.send_cmd(linear_y=-self.speed)
                    self.record_move('right', speed=self.speed)
                    
                elif key == 'q':
                    print(f"🔄 Rotate Left ({self.turn_speed:.2f})")
                    self.send_cmd(angular_z=self.turn_speed)
                    self.record_move('rotate_left', turn_speed=self.turn_speed)
                    
                elif key == 'e':
                    print(f"🔃 Rotate Right ({self.turn_speed:.2f})")
                    self.send_cmd(angular_z=-self.turn_speed)
                    self.record_move('rotate_right', turn_speed=self.turn_speed)
                    
                elif key == 'z':
                    print(f"↖️  Diagonal Forward-Left")
                    self.send_cmd(linear_x=self.speed, linear_y=self.speed)
                    self.record_move('diagonal_fl', speed=self.speed)
                    
                elif key == 'c':
                    print(f"↗️  Diagonal Forward-Right")
                    self.send_cmd(linear_x=self.speed, linear_y=-self.speed)
                    self.record_move('diagonal_fr', speed=self.speed)
                    
                elif key == '+' or key == '=':
                    self.speed = min(0.3, self.speed + 0.05)
                    print(f"⚡ Speed: {self.speed:.2f}")
                    
                elif key == '-':
                    self.speed = max(0.05, self.speed - 0.05)
                    print(f"🐌 Speed: {self.speed:.2f}")
                    
                elif key == '[':
                    self.turn_speed = max(0.1, self.turn_speed - 0.2)
                    turn_time = (2 * math.pi) / self.turn_speed
                    print(f"🔄⬇️  Turn Speed: {self.turn_speed:.2f} rad/s (360° in {turn_time:.1f}s)")
                    
                elif key == ']':
                    self.turn_speed = min(4.0, self.turn_speed + 0.2)
                    turn_time = (2 * math.pi) / self.turn_speed
                    print(f"🔄⬆️  Turn Speed: {self.turn_speed:.2f} rad/s (360° in {turn_time:.1f}s)")
                    
                elif key == 'r':
                    self.full_turn_right()
                    
                elif key == ' ':
                    print("🛑 STOP")
                    self.send_cmd()
                    self.record_move('stop')
                    
                    # If in recording mode, stop recording after space
                    if self.recording_mode:
                        self.stop_recording()
                    
                elif key == 'R':
                    if self.recording_mode:
                        self.stop_recording()
                    else:
                        self.start_recording()
                        
                elif key == 'p':
                    if self.recorded_moves:
                        print("Available moves:")
                        for name in self.recorded_moves.keys():
                            print(f"  - {name}")
                        move_name = input("Enter move name to play: ").strip()
                        if move_name:
                            self.play_move(move_name)
                    else:
                        print("❌ No recorded moves available")
                        
                elif key == 'l':
                    self.list_moves()
                    
                elif key == 'd':
                    if self.recorded_moves:
                        print("Available moves:")
                        for name in self.recorded_moves.keys():
                            print(f"  - {name}")
                        move_name = input("Enter move name to delete: ").strip()
                        if move_name:
                            self.delete_move(move_name)
                    else:
                        print("❌ No recorded moves available")
                    
                elif key == 'x' or key == '\x03':  # x or Ctrl+C
                    print("👋 Exiting...")
                    break
                    
                else:
                    if key:  # Only print for non-empty keys
                        print("❓ Unknown key. Press 'x' to exit.")
                    
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
        finally:
            # Stop robot and restore terminal
            self.send_cmd()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)

def main(args=None):
    # Parse ROS arguments
    rclpy.init(args=args)
    
    # Parse remaining arguments for robot namespace
    import sys
    robot_namespace = ''
    
    # Simple argument parsing for --robot
    if '--robot' in sys.argv:
        try:
            robot_idx = sys.argv.index('--robot')
            if robot_idx + 1 < len(sys.argv):
                robot_namespace = sys.argv[robot_idx + 1]
        except (ValueError, IndexError):
            pass
    
    try:
        controller = MultiRobotTeleop(robot_namespace)
        controller.run()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()