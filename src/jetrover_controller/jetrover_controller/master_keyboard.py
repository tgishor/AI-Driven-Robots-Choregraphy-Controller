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

class MasterRobotTeleop(Node):
    def __init__(self, robot_namespaces=None):
        # Create unique node name
        node_name = 'master_robot_teleop'
        super().__init__(node_name)
        
        # Default to robot1 and robot2 if no namespaces provided
        if robot_namespaces is None:
            robot_namespaces = ['robot1', 'robot2']
        
        self.robot_namespaces = robot_namespaces
        
        # Create publishers for each robot
        self.robot_publishers = {}
        for namespace in robot_namespaces:
            topic_name = f'/{namespace}/controller/cmd_vel'
            self.robot_publishers[namespace] = self.create_publisher(Twist, topic_name, 10)
            print(f"📡 Publisher created for {namespace}: {topic_name}")
        
        self.speed = 0.1
        self.turn_speed = 0.5  # Start conservative, use ] to increase
        
        # Position tracking for each robot
        self.robot_positions = {}
        for namespace in robot_namespaces:
            self.robot_positions[namespace] = {
                'x': 0.0,
                'y': 0.0,
                'theta': 0.0,
                'last_movement_time': time.time()
            }
        
        # Recording functionality
        self.recording_mode = False
        self.current_move_sequence = []
        self.recorded_moves = {}
        self.moves_file = "master_recorded_moves.json"
        self.load_recorded_moves()
        
        # Choreography functionality
        self.choreography_mode = False
        self.choreography_moves = {}
        self.choreography_file = "master_choreography.json"
        self.load_choreography()
        
        print("🎭 Master Multi-Robot Keyboard Teleop")
        print(f"🤖 Controlling Robots: {', '.join(robot_namespaces)}")
        print("="*50)
        print("Controls:")
        print("  w/s: Forward/Backward (all robots)")
        print("  a/d: Left/Right (all robots)")
        print("  q/e: Rotate Left/Right (all robots)")
        print("  z/c: Diagonal movements (all robots)")
        print("  +/-: Increase/Decrease speed")
        print("  [/]: Decrease/Increase turn speed")
        print("  r: Full 360° turn right (test)")
        print("  SPACE: Stop all robots")
        print("  x: Exit")
        print("")
        print("🎬 Recording Controls:")
        print("  R: Start/Stop recording mode")
        print("  p: Play recorded move")
        print("  l: List recorded moves")
        print("  d: Delete recorded move")
        print("")
        print("🎭 Choreography Controls:")
        print("  C: Start/Stop choreography mode")
        print("  t: Create timed movement")
        print("  s: Create spread & align movement")
        print("  A: Create advanced multi-phase movement")
        print("  P: Play choreography")
        print("  L: List choreographies")
        print("  D: Delete choreography")
        print("  i: Show all robot positions")
        print("  0: Reset all positions to origin")
        print("="*50)
        turn_time = (2 * math.pi) / self.turn_speed
        print(f"Current speed: {self.speed:.2f} m/s")
        print(f"Current turn speed: {self.turn_speed:.2f} rad/s (360° in {turn_time:.1f}s)")
        print(f"Publishing to {len(self.robot_publishers)} robots")
        
    def update_position(self, robot_namespace, linear_x, linear_y, angular_z, duration):
        """Update robot position based on movement"""
        if robot_namespace not in self.robot_positions:
            return
            
        pos = self.robot_positions[robot_namespace]
        
        # Update position based on linear movement
        pos['x'] += linear_x * duration
        pos['y'] += linear_y * duration
        
        # Update orientation based on angular movement
        pos['theta'] += angular_z * duration
        
        # Keep theta in [-π, π]
        pos['theta'] = math.atan2(math.sin(pos['theta']), math.cos(pos['theta']))
        
        pos['last_movement_time'] = time.time()
        
    def get_position(self, robot_namespace):
        """Get current robot position"""
        if robot_namespace not in self.robot_positions:
            return None
        return self.robot_positions[robot_namespace].copy()
        
    def reset_all_positions(self):
        """Reset all robot positions to origin"""
        for namespace in self.robot_positions:
            self.robot_positions[namespace] = {
                'x': 0.0,
                'y': 0.0,
                'theta': 0.0,
                'last_movement_time': time.time()
            }
        print("🔄 All robot positions reset to origin (0, 0, 0)")
        
    def show_all_positions(self):
        """Show current positions of all robots"""
        print(f"📍 Current Positions:")
        for namespace in self.robot_namespaces:
            pos = self.get_position(namespace)
            if pos:
                print(f"   {namespace}: X={pos['x']:.3f}m, Y={pos['y']:.3f}m, θ={math.degrees(pos['theta']):.1f}°")
        
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
            
    def load_choreography(self):
        """Load choreography from JSON file"""
        try:
            if os.path.exists(self.choreography_file):
                with open(self.choreography_file, 'r') as f:
                    self.choreography_moves = json.load(f)
                print(f"📂 Loaded {len(self.choreography_moves)} choreographies from {self.choreography_file}")
        except Exception as e:
            print(f"⚠️  Could not load choreography: {e}")
            self.choreography_moves = {}
            
    def save_choreography(self):
        """Save choreography to JSON file"""
        try:
            with open(self.choreography_file, 'w') as f:
                json.dump(self.choreography_moves, f, indent=2)
            print(f"💾 Saved {len(self.choreography_moves)} choreographies to {self.choreography_file}")
        except Exception as e:
            print(f"❌ Error saving choreography: {e}")
            
    def create_advanced_multi_phase_movement(self):
        """Create an advanced multi-phase movement choreography"""
        print("🎯 Creating Advanced Multi-Phase Movement")
        print("This creates a complex movement with primary and secondary phases")
        print("Robots move in opposite directions with position tracking and alignment")
        
        try:
            primary_time = float(input("Primary movement time (seconds): "))
            secondary_time = float(input("Secondary movement time (seconds): "))
            speed = float(input("Speed (m/s): "))
            
            choreography_name = input("Choreography name: ").strip()
            if not choreography_name:
                print("❌ No name provided")
                return
                
            # Create choreography data
            choreography_data = {
                'type': 'advanced_multi_phase',
                'primary_time': primary_time,
                'secondary_time': secondary_time,
                'speed': speed,
                'timestamp': datetime.now().isoformat(),
                'robot_namespaces': self.robot_namespaces
            }
            
            self.choreography_moves[choreography_name] = choreography_data
            self.save_choreography()
            print(f"✅ Created advanced multi-phase movement: {choreography_name}")
            print(f"   Primary: {primary_time}s, Secondary: {secondary_time}s, Speed: {speed} m/s")
            print(f"   Robots: {', '.join(self.robot_namespaces)}")
            
        except ValueError:
            print("❌ Invalid input")
            
    def create_timed_movement(self):
        """Create a timed movement choreography"""
        print("⏱️  Creating Timed Movement")
        print("Enter movement parameters:")
        
        try:
            duration = float(input("Duration (seconds): "))
            speed = float(input("Speed (m/s): "))
            direction = input("Direction (left/right/forward/backward): ").lower()
            
            if direction not in ['left', 'right', 'forward', 'backward']:
                print("❌ Invalid direction")
                return
                
            choreography_name = input("Choreography name: ").strip()
            if not choreography_name:
                print("❌ No name provided")
                return
                
            # Create choreography data
            choreography_data = {
                'type': 'timed_movement',
                'duration': duration,
                'speed': speed,
                'direction': direction,
                'timestamp': datetime.now().isoformat(),
                'robot_namespaces': self.robot_namespaces
            }
            
            self.choreography_moves[choreography_name] = choreography_data
            self.save_choreography()
            print(f"✅ Created timed movement: {choreography_name}")
            
        except ValueError:
            print("❌ Invalid input")
            
    def create_spread_align_movement(self):
        """Create a spread and align movement choreography"""
        print("🔄 Creating Spread & Align Movement")
        print("This creates synchronized movement where robots move in opposite directions")
        print("and then align back to the same vertical line")
        
        try:
            duration = float(input("Movement duration (seconds): "))
            speed = float(input("Speed (m/s): "))
            
            choreography_name = input("Choreography name: ").strip()
            if not choreography_name:
                print("❌ No name provided")
                return
                
            # Create choreography data
            choreography_data = {
                'type': 'spread_align',
                'duration': duration,
                'speed': speed,
                'timestamp': datetime.now().isoformat(),
                'robot_namespaces': self.robot_namespaces
            }
            
            self.choreography_moves[choreography_name] = choreography_data
            self.save_choreography()
            print(f"✅ Created spread & align movement: {choreography_name}")
            print(f"   Duration: {duration}s, Speed: {speed} m/s")
            print(f"   Robots: {', '.join(self.robot_namespaces)}")
            
        except ValueError:
            print("❌ Invalid input")
            
    def play_choreography(self, choreography_name):
        """Play a choreography movement"""
        if choreography_name not in self.choreography_moves:
            print(f"❌ Choreography '{choreography_name}' not found")
            return
            
        choreography = self.choreography_moves[choreography_name]
        choreography_type = choreography['type']
        
        print(f"🎭 Playing choreography: {choreography_name} ({choreography_type})")
        
        if choreography_type == 'timed_movement':
            self.play_timed_movement(choreography)
        elif choreography_type == 'spread_align':
            self.play_spread_align_movement(choreography)
        elif choreography_type == 'advanced_multi_phase':
            self.play_advanced_multi_phase_movement(choreography)
        else:
            print(f"❌ Unknown choreography type: {choreography_type}")
            
    def play_advanced_multi_phase_movement(self, choreography):
        """Play an advanced multi-phase movement"""
        primary_time = choreography['primary_time']
        secondary_time = choreography['secondary_time']
        speed = choreography['speed']
        
        print(f"🎯 Executing advanced multi-phase movement")
        print(f"   Primary: {primary_time}s, Secondary: {secondary_time}s, Speed: {speed} m/s")
        
        # Execute for each robot with opposite movements
        for i, namespace in enumerate(self.robot_namespaces):
            is_robot1 = i == 0  # First robot is robot1
            
            if is_robot1:
                print(f"🤖 {namespace} - Phase 1: RIGHT, Phase 2: LEFT")
                
                # Phase 1: Move RIGHT for primary_time
                print(f"   Phase 1: {namespace} moving RIGHT for {primary_time}s")
                start_time = time.time()
                while time.time() - start_time < primary_time:
                    self.send_cmd_to_robot(namespace, linear_y=-speed)  # Right movement
                    time.sleep(0.1)
                self.update_position(namespace, 0, -speed, 0, primary_time)
                
                # Phase 2: Move LEFT for secondary_time
                print(f"   Phase 2: {namespace} moving LEFT for {secondary_time}s")
                start_time = time.time()
                while time.time() - start_time < secondary_time:
                    self.send_cmd_to_robot(namespace, linear_y=speed)  # Left movement
                    time.sleep(0.1)
                self.update_position(namespace, 0, speed, 0, secondary_time)
                
            else:
                print(f"🤖 {namespace} - Phase 1: LEFT, Phase 2: RIGHT")
                
                # Phase 1: Move LEFT for primary_time
                print(f"   Phase 1: {namespace} moving LEFT for {primary_time}s")
                start_time = time.time()
                while time.time() - start_time < primary_time:
                    self.send_cmd_to_robot(namespace, linear_y=speed)  # Left movement
                    time.sleep(0.1)
                self.update_position(namespace, 0, speed, 0, primary_time)
                
                # Phase 2: Move RIGHT for secondary_time
                print(f"   Phase 2: {namespace} moving RIGHT for {secondary_time}s")
                start_time = time.time()
                while time.time() - start_time < secondary_time:
                    self.send_cmd_to_robot(namespace, linear_y=-speed)  # Right movement
                    time.sleep(0.1)
                self.update_position(namespace, 0, -speed, 0, secondary_time)
        
        # Stop all robots and show final positions
        self.send_cmd_to_all()
        print(f"✅ Completed advanced multi-phase movement")
        
        # Show final positions
        for namespace in self.robot_namespaces:
            final_pos = self.get_position(namespace)
            if final_pos:
                print(f"   {namespace}: x={final_pos['x']:.3f}, y={final_pos['y']:.3f}, θ={math.degrees(final_pos['theta']):.1f}°")
        
        # Calculate and perform alignment for each robot
        for namespace in self.robot_namespaces:
            final_pos = self.get_position(namespace)
            if final_pos and abs(final_pos['y']) > 0.01:  # If not aligned (within 1cm)
                alignment_time = abs(final_pos['y']) / speed
                alignment_direction = -1 if final_pos['y'] > 0 else 1
                print(f"🔄 Aligning {namespace} to vertical line (moving {'left' if alignment_direction > 0 else 'right'} for {alignment_time:.2f}s)")
                
                start_time = time.time()
                while time.time() - start_time < alignment_time:
                    self.send_cmd_to_robot(namespace, linear_y=alignment_direction * speed)
                    time.sleep(0.1)
                self.send_cmd_to_robot(namespace)  # Stop this robot
                self.update_position(namespace, 0, alignment_direction * speed, 0, alignment_time)
                
                final_pos = self.get_position(namespace)
                print(f"✅ {namespace} alignment complete. Final position: x={final_pos['x']:.3f}, y={final_pos['y']:.3f}")
        
        # Final stop for all
        self.send_cmd_to_all()
            
    def play_timed_movement(self, choreography):
        """Play a timed movement"""
        duration = choreography['duration']
        speed = choreography['speed']
        direction = choreography['direction']
        
        print(f"⏱️  Executing {direction} movement for {duration}s at {speed} m/s")
        
        # Determine movement command based on direction
        linear_x, linear_y = 0, 0
        if direction == 'forward':
            linear_x = speed
        elif direction == 'backward':
            linear_x = -speed
        elif direction == 'left':
            linear_y = speed
        elif direction == 'right':
            linear_y = -speed
            
        # Execute movement for all robots
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_cmd_to_all(linear_x=linear_x, linear_y=linear_y)
            time.sleep(0.1)  # 10Hz control loop
            
        # Update positions and stop
        for namespace in self.robot_namespaces:
            self.update_position(namespace, linear_x, linear_y, 0, duration)
        self.send_cmd_to_all()
        print(f"✅ Completed timed movement")
        
    def play_spread_align_movement(self, choreography):
        """Play a spread and align movement"""
        duration = choreography['duration']
        speed = choreography['speed']
        
        print(f"🔄 Executing spread & align movement for {duration}s at {speed} m/s")
        
        # Execute movement for each robot with opposite directions
        for i, namespace in enumerate(self.robot_namespaces):
            is_robot1 = i == 0  # First robot is robot1
            
            if is_robot1:
                # Robot 1 moves left
                linear_y = speed
                print(f"🤖 {namespace} moving LEFT")
            else:
                # Robot 2 moves right
                linear_y = -speed
                print(f"🤖 {namespace} moving RIGHT")
                
            # Execute movement
            start_time = time.time()
            while time.time() - start_time < duration:
                self.send_cmd_to_robot(namespace, linear_y=linear_y)
                time.sleep(0.1)  # 10Hz control loop
                
            # Update position
            self.update_position(namespace, 0, linear_y, 0, duration)
            
        # Stop all and align
        self.send_cmd_to_all()
        print(f"✅ Completed spread & align movement")
        print(f"   Robots should now be aligned at the same vertical line")
        
    def list_choreography(self):
        """List all choreography moves"""
        if not self.choreography_moves:
            print("📋 No choreography moves found")
            return
            
        print("📋 Choreography Moves:")
        print("-" * 60)
        for name, data in self.choreography_moves.items():
            choreography_type = data['type']
            timestamp = data['timestamp']
            if choreography_type == 'timed_movement':
                duration = data['duration']
                speed = data['speed']
                direction = data['direction']
                print(f"  {name}: {choreography_type} - {direction} for {duration}s at {speed} m/s")
            elif choreography_type == 'spread_align':
                duration = data['duration']
                speed = data['speed']
                print(f"  {name}: {choreography_type} - {duration}s at {speed} m/s")
            elif choreography_type == 'advanced_multi_phase':
                primary_time = data['primary_time']
                secondary_time = data['secondary_time']
                speed = data['speed']
                print(f"  {name}: {choreography_type} - Primary: {primary_time}s, Secondary: {secondary_time}s at {speed} m/s")
            print(f"    Created: {timestamp}")
        print("-" * 60)
        
    def delete_choreography(self, choreography_name):
        """Delete a choreography move"""
        if choreography_name in self.choreography_moves:
            del self.choreography_moves[choreography_name]
            self.save_choreography()
            print(f"🗑️  Deleted choreography: {choreography_name}")
        else:
            print(f"❌ Choreography '{choreography_name}' not found")
        
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
        
        # Play the sequence for all robots
        for i, action in enumerate(sequence):
            print(f"  [{i+1}/{len(sequence)}] {action['type']}")
            
            if action['type'] == 'forward':
                self.send_cmd_to_all(linear_x=action['speed'])
            elif action['type'] == 'backward':
                self.send_cmd_to_all(linear_x=-action['speed'])
            elif action['type'] == 'left':
                self.send_cmd_to_all(linear_y=action['speed'])
            elif action['type'] == 'right':
                self.send_cmd_to_all(linear_y=-action['speed'])
            elif action['type'] == 'rotate_left':
                self.send_cmd_to_all(angular_z=action['turn_speed'])
            elif action['type'] == 'rotate_right':
                self.send_cmd_to_all(angular_z=-action['turn_speed'])
            elif action['type'] == 'diagonal_fl':
                self.send_cmd_to_all(linear_x=action['speed'], linear_y=action['speed'])
            elif action['type'] == 'diagonal_fr':
                self.send_cmd_to_all(linear_x=action['speed'], linear_y=-action['speed'])
            elif action['type'] == 'stop':
                self.send_cmd_to_all()
                
            # Wait for the duration (if specified) or a short delay
            duration = action.get('duration', 0.1)
            time.sleep(duration)
            
        # Stop at the end
        self.send_cmd_to_all()
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
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.linear.y = float(linear_y)
        twist.angular.z = float(angular_z)
        
        for publisher in self.robot_publishers.values():
            publisher.publish(twist)
        
    def full_turn_right(self):
        """Execute a full 360° turn to the right for all robots"""
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
                self.send_cmd_to_all(angular_z=-self.turn_speed)  # Negative for right turn
                time.sleep(1.0 / rate)
            
            # Stop all robots
            self.send_cmd_to_all()  # Send stop command
            
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
                    print(f"⬆️  Forward ({self.speed:.2f}) - All robots")
                    self.send_cmd_to_all(linear_x=self.speed)
                    self.record_move('forward', speed=self.speed)
                    
                elif key == 's':
                    print(f"⬇️  Backward ({self.speed:.2f}) - All robots")
                    self.send_cmd_to_all(linear_x=-self.speed)
                    self.record_move('backward', speed=self.speed)
                    
                elif key == 'a':
                    print(f"⬅️  Left ({self.speed:.2f}) - All robots")
                    self.send_cmd_to_all(linear_y=self.speed)
                    self.record_move('left', speed=self.speed)
                    
                elif key == 'd':
                    print(f"➡️  Right ({self.speed:.2f}) - All robots")
                    self.send_cmd_to_all(linear_y=-self.speed)
                    self.record_move('right', speed=self.speed)
                    
                elif key == 'q':
                    print(f"🔄 Rotate Left ({self.turn_speed:.2f}) - All robots")
                    self.send_cmd_to_all(angular_z=self.turn_speed)
                    self.record_move('rotate_left', turn_speed=self.turn_speed)
                    
                elif key == 'e':
                    print(f"🔃 Rotate Right ({self.turn_speed:.2f}) - All robots")
                    self.send_cmd_to_all(angular_z=-self.turn_speed)
                    self.record_move('rotate_right', turn_speed=self.turn_speed)
                    
                elif key == 'z':
                    print(f"↖️  Diagonal Forward-Left - All robots")
                    self.send_cmd_to_all(linear_x=self.speed, linear_y=self.speed)
                    self.record_move('diagonal_fl', speed=self.speed)
                    
                elif key == 'c':
                    print(f"↗️  Diagonal Forward-Right - All robots")
                    self.send_cmd_to_all(linear_x=self.speed, linear_y=-self.speed)
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
                    print("🛑 STOP - All robots")
                    self.send_cmd_to_all()
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
                        
                # Choreography controls
                elif key == 'C':
                    if self.choreography_mode:
                        self.choreography_mode = False
                        print("🎭 Choreography mode DISABLED")
                    else:
                        self.choreography_mode = True
                        print("🎭 Choreography mode ENABLED")
                        
                elif key == 't':
                    self.create_timed_movement()
                    
                elif key == 's':
                    self.create_spread_align_movement()
                    
                elif key == 'A':
                    self.create_advanced_multi_phase_movement()
                    
                elif key == 'P':
                    if self.choreography_moves:
                        print("Available choreographies:")
                        for name in self.choreography_moves.keys():
                            print(f"  - {name}")
                        choreography_name = input("Enter choreography name to play: ").strip()
                        if choreography_name:
                            self.play_choreography(choreography_name)
                    else:
                        print("❌ No choreographies available")
                        
                elif key == 'L':
                    self.list_choreography()
                    
                elif key == 'D':
                    if self.choreography_moves:
                        print("Available choreographies:")
                        for name in self.choreography_moves.keys():
                            print(f"  - {name}")
                        choreography_name = input("Enter choreography name to delete: ").strip()
                        if choreography_name:
                            self.delete_choreography(choreography_name)
                    else:
                        print("❌ No choreographies available")
                        
                elif key == 'i':
                    self.show_all_positions()
                    
                elif key == '0':
                    self.reset_all_positions()
                    
                elif key == 'x' or key == '\x03':  # x or Ctrl+C
                    print("👋 Exiting...")
                    break
                    
                else:
                    if key:  # Only print for non-empty keys
                        print("❓ Unknown key. Press 'x' to exit.")
                    
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
        finally:
            # Stop all robots and restore terminal
            self.send_cmd_to_all()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)

def main(args=None):
    # Parse ROS arguments
    rclpy.init(args=args)
    
    # Parse remaining arguments for robot namespaces
    import sys
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
        controller = MasterRobotTeleop(robot_namespaces)
        controller.run()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
