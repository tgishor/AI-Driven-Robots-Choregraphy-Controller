#!/usr/bin/env python3
import os
import time
import threading
import subprocess
import argparse
import random
import json
import pickle
from collections import defaultdict
import rclpy
from rclpy.node import Node
from servo_controller_msgs.msg import ServosPosition, ServoPosition, ServoStateList
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

import librosa
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import sys
import termios
import tty
import select

default_audio = '/home/ubuntu/axy_proj/src/jetrover_controller/jetrover_controller/On-The-Floor.mp3'

class AdvancedDanceNode(Node):
    def __init__(self, audio_path, audio_player='mpg123', buffer_time=2.0, energy_scale=0.30):
        super().__init__('advanced_dance_node')
        self.audio_path = audio_path
        self.audio_player = audio_player
        self.buffer_time = buffer_time  # Buffer to prevent mid-performance stops
        self.energy_scale = energy_scale  # Scale factor for movement intensity (0.1-1.0)

        if not os.path.isfile(self.audio_path):
            self.get_logger().error(f"Audio file not found: {self.audio_path}")
            raise FileNotFoundError(self.audio_path)

        # Publishers and subscribers - TRIPLE ROBOT SUPPORT
        self.servo_pub_robot1 = self.create_publisher(ServosPosition, '/robot_1/servo_controller', 10)
        self.servo_pub_robot2 = self.create_publisher(ServosPosition, '/robot_2/servo_controller', 10)
        self.servo_pub_robot3 = self.create_publisher(ServosPosition, '/robot_3/servo_controller', 10)
        self.cmd_vel_pub_robot1 = self.create_publisher(Twist, '/robot_1/controller/cmd_vel', 10)
        self.cmd_vel_pub_robot2 = self.create_publisher(Twist, '/robot_2/controller/cmd_vel', 10)
        self.cmd_vel_pub_robot3 = self.create_publisher(Twist, '/robot_3/controller/cmd_vel', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/dance/emergency_stop', 10)
        
        self.current_pose = {}
        self.pose_received = False
        self.create_subscription(ServoStateList, '/robot_2/controller_manager/servo_states', self.servo_state_cb, 10)
        self.create_subscription(Bool, '/dance/stop_command', self.emergency_stop_cb, 10)

        # Robot configuration
        self.servo_ids = [1, 2, 3, 4, 5, 10]
        self.max_servo_delta = 400.0
        self.home_positions = {sid: 500 for sid in self.servo_ids}
        
        # Performance control
        self.performance_active = False
        self.emergency_stop_requested = False
        self.audio_process = None
        
        # Pre-planned choreography
        self.choreography_timeline = []
        self.movement_signatures = {}
        self.musical_features = {}
        
        # Movement tracking to prevent continuous forward movement
        self.movement_counter = 0
        self.accumulated_displacement = {'x': 0.0, 'y': 0.0}
        self.last_forced_return = 0
        
        # STRICT 3-SECOND FORWARD LIMIT ENFORCEMENT
        self.forward_start_time = None
        self.total_forward_time = 0.0
        self.is_currently_moving_forward = False
        self.last_movement_check_time = time.time()
        self.FORWARD_TIME_LIMIT = 3.0  # MAXIMUM 3 seconds forward before FORCED backward
        
        # Keyboard monitoring for 'S' key stop
        self.keyboard_monitoring = False
        self.original_settings = None
        
        # Robot orientation tracking for smooth returns
        self.current_orientation = 0.0  # Track total rotation in radians
        self.orientation_history = []
        self.last_spin_direction = None
        self.spin_completion_time = {}  # Track time needed for each spin type
        
        # Robot_3 specific movement tracking (different mechanism)
        self.robot3_move_direction = -1  # Alternates between forward/backward for robot_3
        self.beat_counter = 0  # Track beats for robot_3's every-8-beats pattern
        
        # Enhanced movement vocabulary including Mecanum-specific movements
        self.movement_types = {
            # Arm-based servo movements
            'gentle_wave': {'energy': 'low', 'brightness': 'low', 'pattern': 'smooth', 'type': 'servo'},
            'energetic_wave': {'energy': 'high', 'brightness': 'high', 'pattern': 'sharp', 'type': 'servo'},
            'deep_pulse': {'energy': 'medium', 'brightness': 'low', 'pattern': 'rhythmic', 'type': 'servo'},
            'bright_sparkle': {'energy': 'high', 'brightness': 'high', 'pattern': 'quick', 'type': 'servo'},
            'flowing_reach': {'energy': 'medium', 'brightness': 'medium', 'pattern': 'sustained', 'type': 'servo'},
            'dramatic_sweep': {'energy': 'high', 'brightness': 'medium', 'pattern': 'dramatic', 'type': 'servo'},
            'subtle_sway': {'energy': 'low', 'brightness': 'medium', 'pattern': 'gentle', 'type': 'servo'},
            'powerful_strike': {'energy': 'very_high', 'brightness': 'high', 'pattern': 'accent', 'type': 'servo'},
            
            # PURE SPINNING movements - COMPLETE ROTATIONS with calculated durations
            'pause': {'energy': 'none', 'brightness': 'none', 'pattern': 'stationary', 'type': 'base'},  # NEW: No movement pause
            'return_to_origin': {'energy': 'low', 'brightness': 'low', 'pattern': 'return', 'type': 'base'},  # NEW: Return to start orientation
            'quarter_turn_left': {'energy': 'low', 'brightness': 'low', 'pattern': 'quarter_rotation_left', 'type': 'base', 'rotation': 1.571},  # 90 degrees
            'quarter_turn_right': {'energy': 'low', 'brightness': 'low', 'pattern': 'quarter_rotation_right', 'type': 'base', 'rotation': -1.571},
            'half_spin_left': {'energy': 'medium', 'brightness': 'medium', 'pattern': 'half_rotation_left', 'type': 'base', 'rotation': 3.142},  # 180 degrees
            'half_spin_right': {'energy': 'medium', 'brightness': 'medium', 'pattern': 'half_rotation_right', 'type': 'base', 'rotation': -3.142},
            'full_spin_left': {'energy': 'high', 'brightness': 'high', 'pattern': 'full_rotation_left', 'type': 'base', 'rotation': 6.283},  # 360 degrees
            'full_spin_right': {'energy': 'high', 'brightness': 'high', 'pattern': 'full_rotation_right', 'type': 'base', 'rotation': -6.283},
            'double_spin_left': {'energy': 'very_high', 'brightness': 'very_high', 'pattern': 'double_rotation_left', 'type': 'base', 'rotation': 12.566},  # 720 degrees
            'double_spin_right': {'energy': 'very_high', 'brightness': 'very_high', 'pattern': 'double_rotation_right', 'type': 'base', 'rotation': -12.566}
        }
        
        self.get_logger().info(f"Starting comprehensive music analysis...")
        self.get_logger().info(f"🎚️ Energy Scale: {self.energy_scale:.2f} (lower = gentler movements)")
        self.get_logger().info(f"🤖🤖🤖 TRIPLE ROBOT MODE: Commands will be sent to robot_1, robot_2, and robot_3!")
        self.get_logger().info(f"💃 BALANCED DANCE MODE: Prioritizing expressive arm movements with occasional spins!")
        self.get_logger().info(f"🎭 ARM FOCUS: 70-90% arm movements for maximum expression!")
        self.get_logger().info(f"🌀 ROBOT_1&2: Gentle rotations within 50cm space when music peaks!")
        self.get_logger().info(f"🏃 ROBOT_3: Amplitude-based linear movement (forward/backward) every 8 beats!")
        self.get_logger().info(f"📍 ORIENTATION TRACKING: Robot returns to original facing direction!")
        self.get_logger().info(f"🛑 Multiple Stop Methods: Ctrl+C OR press 'S' key - 1 stop command per second!")
        self.get_logger().info(f"🎨 EXPRESSIVE CHOREOGRAPHY: Beautiful arm movements with strategic movements!")
        # Complete analysis BEFORE starting any performance
        self.analyze_complete_song()
        self.create_choreography_plan()
        self.get_logger().info("Choreography fully planned and ready for flawless triple robot hybrid dancing!")

    def servo_state_cb(self, msg: ServoStateList):
        for s in msg.servo_state:
            self.current_pose[s.id] = s.position
        self.pose_received = True

    def emergency_stop_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().warn("🚨 EMERGENCY STOP MESSAGE RECEIVED!")
            self.emergency_stop()

    def analyze_complete_song(self):
        """Complete musical analysis with comprehensive feature extraction"""
        self.get_logger().info("Performing deep musical analysis...")
        
        try:
            # Load audio with error handling
            y, sr = librosa.load(self.audio_path, sr=22050)  # Standard sample rate for efficiency
            self.song_duration = len(y) / sr
            
            # Extract all musical features
            self.musical_features = {
                'audio': y,
                'sample_rate': sr,
                'duration': self.song_duration
            }
            
            # Rhythm analysis
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Ensure tempo is a scalar value
            tempo_value = float(tempo) if hasattr(tempo, '__len__') and len(tempo) > 0 else float(tempo)
            
            self.musical_features.update({
                'tempo': tempo_value,
                'beats': beats,
                'beat_times': beat_times
            })
            
            # Energy and dynamics
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            rms_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            self.musical_features.update({
                'onset_strength': onset_env,
                'rms_energy': rms_energy,
                'energy_median': np.median(rms_energy)
            })
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            self.musical_features.update({
                'spectral_centroids': spectral_centroids,
                'spectral_rolloff': spectral_rolloff,
                'mfccs': mfccs,
                'chroma': chroma,
                'brightness_median': np.median(spectral_centroids)
            })
            
            # Tonal analysis
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            self.musical_features.update({
                'tonnetz': tonnetz,
                'zero_crossing_rate': zero_crossing_rate
            })
            
            self.get_logger().info(f"Musical analysis complete: {self.song_duration:.1f}s song, {tempo_value:.1f} BPM, {len(beat_times)} beats")
            
        except Exception as e:
            self.get_logger().error(f"Musical analysis failed: {e}")
            raise e

    def create_choreography_plan(self):
        """Create complete choreography plan based on musical analysis"""
        self.get_logger().info("Creating intelligent choreography plan...")
        
        # Get beat times and features
        beat_times = self.musical_features['beat_times']
        
        # Analyze each musical segment - BUT ONLY WITHIN SONG DURATION!
        segments = []
        for i in range(len(beat_times) - 1):
            start_time = beat_times[i]
            end_time = beat_times[i + 1]
            
            # 🔥 CRITICAL FIX: Skip beats that start beyond actual song duration
            if start_time >= self.song_duration:
                self.get_logger().info(f"⏭️ Skipping beat at {start_time:.1f}s - beyond song duration ({self.song_duration:.1f}s)")
                break
            
            # 🔥 CRITICAL FIX: Limit end_time to actual song duration
            if end_time > self.song_duration:
                end_time = self.song_duration
                self.get_logger().info(f"✂️ Trimming final beat to song duration: {end_time:.1f}s")
            
            duration = end_time - start_time
            
            # Skip very short segments
            if duration < 0.05:
                continue
            
            # Extract features for this segment
            segment_features = self.extract_segment_features(start_time, end_time)
            
            # Determine movement type based on musical characteristics
            movement_type = self.classify_movement_type(segment_features)
            
            # For spin movements, ensure they have enough time to complete
            movement_info = self.movement_types.get(movement_type, {})
            if movement_info.get('type') == 'base' and 'rotation' in movement_info:
                # Calculate needed duration for complete rotation
                rotation_amount = abs(movement_info['rotation'])
                angular_speed = 1.0  # rad/s base speed
                needed_duration = rotation_amount / angular_speed
                
                # If the beat is too short for the spin, extend duration
                if duration < needed_duration:
                    duration = needed_duration
                    end_time = start_time + duration
            
            # Calculate movement commands (servo + base)
            movement_commands = self.calculate_movement_commands(movement_type, segment_features)
            
            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'movement_type': movement_type,
                'movement_commands': movement_commands,
                'features': segment_features
            })
            
            # Track orientation for spins
            if movement_info.get('type') == 'base' and movement_type != 'pause':
                if movement_type == 'return_to_origin':
                    self.current_orientation = 0.0
                elif 'rotation' in movement_info:
                    self.current_orientation += movement_info['rotation']
            
            # Periodically add return to origin movements
            if i > 0 and i % 8 == 0 and abs(self.current_orientation) > 0.5:
                # Add a return to origin movement
                return_start = end_time
                return_duration = min(1.0, abs(self.current_orientation) / 1.0)
                return_end = return_start + return_duration
                
                segments.append({
                    'start_time': return_start,
                    'end_time': return_end,
                    'duration': return_duration,
                    'movement_type': 'return_to_origin',
                    'movement_commands': self.calculate_movement_commands('return_to_origin', segment_features),
                    'features': segment_features
                })
                
                self.current_orientation = 0.0
        
        # Smooth transitions and optimize timing
        self.choreography_timeline = self.optimize_choreography(segments)
        
        # Log the actual timeline range
        if self.choreography_timeline:
            first_movement = self.choreography_timeline[0]['start_time']
            last_movement = self.choreography_timeline[-1]['end_time']
            self.get_logger().info(f"📋 Choreography planned: {len(self.choreography_timeline)} movements")
            self.get_logger().info(f"⏱️ Timeline: {first_movement:.1f}s to {last_movement:.1f}s (Song: {self.song_duration:.1f}s)")
            
            # Verify no movements exceed song duration
            if last_movement > self.song_duration:
                self.get_logger().error(f"🚨 ERROR: Movements extend beyond song! Last: {last_movement:.1f}s, Song: {self.song_duration:.1f}s")
            else:
                self.get_logger().info(f"✅ All movements within song duration!")
        else:
            self.get_logger().warn("⚠️ No choreography movements created!")

    def extract_segment_features(self, start_time, end_time):
        """Extract musical features for a specific time segment"""
        sr = self.musical_features['sample_rate']
        start_frame = int(start_time * sr / 512)  # Convert to frame index
        end_frame = int(end_time * sr / 512)
        
        # Extract segment-specific features
        energy = np.mean(self.musical_features['rms_energy'][start_frame:end_frame])
        brightness = np.mean(self.musical_features['spectral_centroids'][start_frame:end_frame])
        onset_strength = np.mean(self.musical_features['onset_strength'][start_frame:end_frame])
        
        # Normalize features
        energy_norm = energy / self.musical_features['energy_median']
        brightness_norm = brightness / self.musical_features['brightness_median']
        
        return {
            'energy': energy_norm,
            'brightness': brightness_norm,
            'onset_strength': onset_strength,
            'duration': end_time - start_time
        }

    def classify_movement_type(self, features):
        """Classify movement type - BALANCED ARM AND SPIN MOVEMENTS for 50cm constraint space"""
        energy = features['energy']
        brightness = features['brightness']
        onset = features['onset_strength']
        duration = features['duration']
        
        # BALANCED CHOREOGRAPHY - Prioritize arm movements, add spins for variety
        # More arm movements to match the Copy file's behavior
        
        # EXPLOSIVE ENERGY - Strong movements
        if energy > 1.8 and brightness > 1.5:
            # 70% arms, 30% spins for high energy
            if onset > 0.8 and random.random() < 0.3:
                return 'double_spin_left' if brightness > energy else 'double_spin_right'  # Base: energetic double spins
            else:
                return 'powerful_strike'  # Servo: maximum intensity arm movement (PRIORITIZED)
                
        # VERY HIGH ENERGY - Dramatic movements
        elif energy > 1.5 and brightness > 1.5:
            # 75% arms, 25% spins
            if onset > 0.7 and random.random() < 0.25:
                return 'full_spin_left' if brightness > energy else 'full_spin_right'  # Base: full spins
            else:
                return 'dramatic_sweep'  # Servo: dramatic arm sweeps (PRIORITIZED)
        
        # HIGH ENERGY - Dynamic movements
        elif energy > 1.2 and brightness > 1.2:
            # 80% arms, 20% spins
            if onset > 0.6 and random.random() < 0.2:
                return 'half_spin_left' if brightness > 1.3 else 'half_spin_right'  # Base: half spins
            else:
                return 'energetic_wave'  # Servo: sharp arm movements (PRIORITIZED)
                
        elif energy > 1.2:
            # Mostly arm movements with occasional spins
            if onset > 0.7 and random.random() < 0.15:
                return 'quarter_turn_left' if brightness > energy else 'quarter_turn_right'  # Base: quarter turns
            else:
                return 'bright_sparkle'  # Servo: quick bright arm movements (PRIORITIZED)
        
        # MEDIUM-HIGH ENERGY - Flowing movements
        elif energy > 1.0 and brightness > 1.2:
            # 85% arms, 15% spins
            if onset > 0.6 and random.random() < 0.15:
                return 'quarter_turn_left' if onset > 0.5 else 'quarter_turn_right'  # Base: gentle turns
            else:
                return 'flowing_reach'  # Servo: sustained arm movements (PRIORITIZED)
                
        elif energy > 1.0 and brightness < 0.8:
            # Deep movements - mostly arms
            if onset > 0.7 and random.random() < 0.1:
                return 'pause'  # Base: pause for emphasis
            else:
                return 'deep_pulse'  # Servo: rhythmic arm pulses (PRIORITIZED)
                
        elif energy > 1.0:
            # Varied movements - mostly arms
            return 'dramatic_sweep'  # Servo: dramatic arm sweeps (PRIORITIZED)
        
        # MEDIUM ENERGY - Gentle variety
        elif brightness > 1.3:
            # Bright, gentle movements - mostly arms
            if onset > 0.5 and random.random() < 0.1:
                return 'quarter_turn_right'  # Base: occasional gentle turn
            else:
                return 'bright_sparkle'  # Servo: quick bright arm movements (PRIORITIZED)
                
        elif energy > 0.8 and brightness > 0.8:
            # Balanced energy - mostly arms
            return 'flowing_reach'  # Servo: sustained arm movements (PRIORITIZED)
        
        # LOWER ENERGY - Gentle arms only
        elif brightness > 1.0:
            return 'subtle_sway'  # Servo: gentle arm sway (ARMS ONLY)
        else:
            return 'gentle_wave'  # Servo: soft arm wave (ARMS ONLY)

    def calculate_movement_commands(self, movement_type, features):
        """Calculate movement commands for both servo and base movements"""
        # Base amplitude from musical energy with scaling control
        scaled_energy = features['energy'] * self.energy_scale
        base_amplitude = min(scaled_energy * self.max_servo_delta, self.max_servo_delta)
        
        # Add TEMPO-based speed scaling for better music synchronization
        tempo = self.musical_features.get('tempo', 120.0)  # Default 120 BPM
        tempo_scale = self.calculate_tempo_speed_scale(tempo)
        
        # Determine if this is a servo or base movement
        movement_info = self.movement_types.get(movement_type, {'type': 'servo'})
        movement_category = movement_info['type']
        
        result = {
            'movement_type': movement_type,
            'category': movement_category,
            'servo_positions': {},
            'base_command': {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0},
            'tempo_scale': tempo_scale  # Pass tempo scaling to movement calculations
        }
        
        if movement_category == 'servo':
            result['servo_positions'] = self.calculate_servo_positions(movement_type, features, base_amplitude)
            # Add subtle base movement to complement servo movements for hybrid dancing
            result['base_command'] = self.calculate_subtle_base_complement(movement_type, features, tempo_scale)
            
        elif movement_category == 'base':
            result['base_command'] = self.calculate_base_movement(movement_type, features, tempo_scale)
            # Also add complementary servo movements to base movement
            result['servo_positions'] = self.calculate_complementary_servo_positions(movement_type, features, base_amplitude)
            
        return result

    def calculate_tempo_speed_scale(self, tempo):
        """Calculate speed scaling factor based on musical tempo"""
        # Scale movement speed based on BPM for better synchronization
        # Faster songs = faster movements, slower songs = more controlled movements
        
        if tempo < 80:
            # Very slow songs (ballads) - controlled movements
            return 0.6
        elif tempo < 100:
            # Slow songs - moderate movements  
            return 0.8
        elif tempo < 120:
            # Medium tempo - normal speed
            return 1.0
        elif tempo < 140:
            # Upbeat songs - faster movements
            return 1.3
        elif tempo < 160:
            # Fast songs - very dynamic
            return 1.6
        else:
            # Very fast songs (EDM, etc.) - MAXIMUM ENERGY!
            return 2.0
    
    def calculate_return_to_origin_command(self, features):
        """Calculate command to return robot to original orientation"""
        # Base amplitude from musical energy for arm movements during return
        scaled_energy = features['energy'] * self.energy_scale
        base_amplitude = min(scaled_energy * self.max_servo_delta, self.max_servo_delta)
        
        result = {
            'movement_type': 'return_to_origin',
            'category': 'base',
            'servo_positions': {},
            'base_command': {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}
        }
        
        # Calculate rotation needed to return to origin
        if abs(self.current_orientation) > 0.1:
            # Fixed return speed for smooth motion
            angular_speed = 1.0  # rad/s
            if self.current_orientation > 0:
                result['base_command']['angular_z'] = -angular_speed
            else:
                result['base_command']['angular_z'] = angular_speed
                
        # Add gentle arm movements during return
        result['servo_positions'] = self.calculate_complementary_servo_positions('return_to_origin', features, base_amplitude)
        
        return result

    def calculate_servo_positions(self, movement_type, features, base_amplitude):
        """Calculate precise servo positions for servo-focused movements"""
        positions = {}
        
        if movement_type == 'gentle_wave':
            for sid in self.servo_ids:
                positions[sid] = self.home_positions[sid] + ((-1) ** sid) * base_amplitude * 0.3
                
        elif movement_type == 'energetic_wave':
            for sid in self.servo_ids:
                positions[sid] = self.home_positions[sid] + ((-1) ** sid) * base_amplitude * 0.7
                
        elif movement_type == 'deep_pulse':
            for sid in self.servo_ids:
                positions[sid] = self.home_positions[sid] + random.uniform(-0.4, 0.4) * base_amplitude
                
        elif movement_type == 'bright_sparkle':
            for sid in self.servo_ids:
                positions[sid] = self.home_positions[sid] + random.uniform(-0.6, 0.6) * base_amplitude
                
        elif movement_type == 'flowing_reach':
            for sid in self.servo_ids:
                reach_direction = 1 if sid % 2 == 0 else -1
                positions[sid] = self.home_positions[sid] + reach_direction * base_amplitude * 0.5
                
        elif movement_type == 'dramatic_sweep':
            for sid in self.servo_ids:
                sweep_pattern = ((-1) ** (sid + 1)) * base_amplitude * 0.8
                positions[sid] = self.home_positions[sid] + sweep_pattern
                
        elif movement_type == 'subtle_sway':
            for sid in self.servo_ids:
                sway = base_amplitude * 0.2 * (1 if sid in [1, 3, 5] else -1)
                positions[sid] = self.home_positions[sid] + sway
                
        elif movement_type == 'powerful_strike':
            for sid in self.servo_ids:
                strike_intensity = base_amplitude * 0.9 * ((-1) ** sid)
                positions[sid] = self.home_positions[sid] + strike_intensity
        
        # Ensure all positions are within safe servo limits
        for sid in positions:
            positions[sid] = max(150, min(900, positions[sid]))
            
        return positions

    def calculate_base_movement(self, movement_type, features, tempo_scale=1.0):
        """Calculate SMOOTH SPINNING movements - 50cm constraint space - NO LINEAR MOVEMENT"""
        energy = features['energy']
        duration = features['duration']
        brightness = features['brightness']
        onset = features['onset_strength']
        
        # Base angular speed for smooth, complete rotations
        base_angular_speed = 1.0  # rad/s - fixed speed for predictable spins
        
        # Small adjustments based on energy
        if energy > 1.5:
            base_angular_speed *= 1.1  # Slightly faster for high energy
        elif energy < 0.8:
            base_angular_speed *= 0.9  # Slightly slower for low energy
        
        # Apply TEMPO SCALING for music synchronization
        angular_speed = base_angular_speed * tempo_scale
        
        # Cap the speed at maximum after tempo scaling
        angular_speed = min(1.5, angular_speed)  # Max 1.5 rad/s total
        
        # 🚨 CRITICAL: ZERO LINEAR MOVEMENT - PURE SPINNING ONLY! 🚨
        # This GUARANTEES robot stays in 50cm constraint space!
        base_command = {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}
        
        # Handle special movement types
        if movement_type == 'pause':
            # No movement - pause for emphasis
            return base_command
            
        elif movement_type == 'return_to_origin':
            # Calculate rotation needed to return to origin
            if abs(self.current_orientation) > 0.1:
                # Rotate back to origin at fixed speed
                return_speed = 1.0  # rad/s
                if self.current_orientation > 0:
                    base_command['angular_z'] = -return_speed
                else:
                    base_command['angular_z'] = return_speed
            return base_command
        
        # Handle standard spin movements
        elif movement_type == 'quarter_turn_left':
            # Quarter turn left (90 degrees)
            base_command['angular_z'] = angular_speed
            
        elif movement_type == 'quarter_turn_right':
            # Quarter turn right (90 degrees)
            base_command['angular_z'] = -angular_speed
            
        elif movement_type == 'half_spin_left':
            # Half spin left (180 degrees)
            base_command['angular_z'] = angular_speed
            
        elif movement_type == 'half_spin_right':
            # Half spin right (180 degrees)
            base_command['angular_z'] = -angular_speed
            
        elif movement_type == 'full_spin_left':
            # Full spin left (360 degrees)
            base_command['angular_z'] = angular_speed
            
        elif movement_type == 'full_spin_right':
            # Full spin right (360 degrees)
            base_command['angular_z'] = -angular_speed
            
        elif movement_type == 'double_spin_left':
            # Double spin left (720 degrees)
            base_command['angular_z'] = angular_speed * 1.2  # Slightly faster for emphasis
            
        elif movement_type == 'double_spin_right':
            # Double spin right (720 degrees)
            base_command['angular_z'] = -angular_speed * 1.2  # Slightly faster for emphasis
            
        return base_command

    def calculate_subtle_base_complement(self, movement_type, features, tempo_scale=1.0):
        """Calculate subtle SPINNING movements to complement servo arm movements - 50cm constraint"""
        energy = features['energy']
        onset = features['onset_strength']
        
        # Subtle SPINNING ONLY - NO LINEAR MOVEMENT for arm complementing
        # VERY GENTLE complement speeds - should barely be noticeable
        angular_speed = min(0.8, energy * 0.3)  # Much gentler rotation for arm complement
        
        # Apply tempo scaling
        angular_speed *= tempo_scale
        
        # Cap the speeds - very low for subtle complement
        angular_speed = min(0.8, angular_speed)  # Max 0.8 rad/s for subtle movements
        
        # 🚨 CRITICAL: ZERO LINEAR MOVEMENT - ONLY SUBTLE SPINS! 🚨
        # This GUARANTEES robot stays in 50cm constraint space!
        base_command = {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}
        
        # Add very subtle SPINNING movements based on arm movement type
        if movement_type in ['powerful_strike', 'dramatic_sweep']:
            # Very subtle rotation during dramatic arm movements
            direction = 1 if onset > 0.5 else -1
            base_command['angular_z'] = direction * angular_speed * 0.2  # Much more subtle
            
        elif movement_type in ['energetic_wave', 'bright_sparkle']:
            # Barely noticeable spins during energetic arm movements
            direction = 1 if energy > 1.0 else -1
            base_command['angular_z'] = direction * angular_speed * 0.15  # Very gentle
            
        elif movement_type in ['flowing_reach', 'subtle_sway']:
            # Extremely gentle rotation during flowing arms
            direction = 1 if onset > 0.4 else -1
            base_command['angular_z'] = direction * angular_speed * 0.1  # Minimal movement
            
        elif movement_type == 'deep_pulse':
            # Very slight rotation to match arm pulses
            direction = 1 if onset > 0.5 else -1
            base_command['angular_z'] = direction * angular_speed * 0.12  # Reduced
                
        elif movement_type == 'gentle_wave':
            # Almost imperceptible rotation to complement gentle arm waves
            direction = 1 if features['brightness'] > 1.0 else -1
            base_command['angular_z'] = direction * angular_speed * 0.08  # Barely noticeable
            
        return base_command

    def calculate_complementary_servo_positions(self, movement_type, features, base_amplitude):
        """Calculate subtle servo movements to complement SPINNING base movements"""
        positions = {}
        
        # Subtle servo movements that complement spinning motions
        complement_amplitude = base_amplitude * 0.3  # Much gentler than main servo movements
        
        for sid in self.servo_ids:
            if movement_type in ['quarter_turn_left', 'quarter_turn_right']:
                # Gentle wave during quarter turns
                positions[sid] = self.home_positions[sid] + ((-1) ** sid) * complement_amplitude * 0.3
            elif movement_type in ['half_spin_left', 'half_spin_right']:
                # Medium reach during half spins
                positions[sid] = self.home_positions[sid] + complement_amplitude * 0.4
            elif movement_type in ['full_spin_left', 'full_spin_right']:
                # Outward reach during full spins
                positions[sid] = self.home_positions[sid] + complement_amplitude * 0.5
            elif movement_type in ['double_spin_left', 'double_spin_right']:
                # Dynamic reach during double spins
                positions[sid] = self.home_positions[sid] + complement_amplitude * 0.6
            elif movement_type == 'pause':
                # Hold neutral position during pause
                positions[sid] = self.home_positions[sid]
            elif movement_type == 'return_to_origin':
                # Gentle return to neutral during orientation return
                positions[sid] = self.home_positions[sid] + random.uniform(-0.1, 0.1) * complement_amplitude
            else:
                # Default: minimal movement
                positions[sid] = self.home_positions[sid] + random.uniform(-0.1, 0.1) * complement_amplitude
        
        # Ensure all positions are within safe servo limits
        for sid in positions:
            positions[sid] = max(150, min(900, positions[sid]))
            
        return positions

    def optimize_choreography(self, segments):
        """Optimize choreography for smooth execution"""
        optimized = []
        
        for i, segment in enumerate(segments):
            # Add buffer time for smooth transitions    
            if i > 0:
                prev_segment = segments[i-1]
                # Ensure smooth transition between movements
                segment['transition_time'] = 0.1
            else:
                segment['transition_time'] = 0.0
                
            optimized.append(segment)
        
        return optimized

    def has_linear_movement(self, movement_commands):
        """Check if movement has any linear displacement that needs return"""
        base_command = movement_commands.get('base_command', {})
        linear_x = abs(base_command.get('linear_x', 0.0))
        linear_y = abs(base_command.get('linear_y', 0.0))
        
        # Check if there's ANY linear movement (very low threshold for return movement)
        return linear_x > 0.05 or linear_y > 0.05

    def create_return_movement(self, original_commands, features):
        """Create return movement to cancel out displacement"""
        base_command = original_commands.get('base_command', {})
        
        # Create exact opposite movement to return to original position
        return_base_command = {
            'linear_x': -base_command.get('linear_x', 0.0),  # Opposite direction
            'linear_y': -base_command.get('linear_y', 0.0),  # Opposite direction
            'angular_z': 0.0  # Don't reverse spins - they're fine as-is
        }
        
        # Create return movement for ANY linear displacement (very low threshold)
        if abs(return_base_command['linear_x']) > 0.05 or abs(return_base_command['linear_y']) > 0.05:
            return {
                'movement_type': 'return_movement',
                'category': 'base',
                'servo_positions': {},  # No servo movement during return
                'base_command': return_base_command
            }
        
        return None

    def create_forced_return_to_center(self):
        """Create movement to return robot to center position"""
        # Calculate return movement based on accumulated displacement
        return_base_command = {
            'linear_x': -self.accumulated_displacement['x'] * 2.0,  # Double strength return
            'linear_y': -self.accumulated_displacement['y'] * 2.0,  # Double strength return
            'angular_z': 0.0  # No rotation
        }
        
        # Cap return speeds to safe levels
        return_base_command['linear_x'] = max(-0.8, min(0.8, return_base_command['linear_x']))
        return_base_command['linear_y'] = max(-0.8, min(0.8, return_base_command['linear_y']))
        
        return {
            'movement_type': 'forced_return_to_center',
            'category': 'base',
            'servo_positions': {},  # No servo movement during return
            'base_command': return_base_command
        }

    def update_forward_movement_tracking(self, base_command, duration):
        """Track forward movement time and enforce 3-second limit - STRICT ENFORCEMENT"""
        current_time = time.time()
        linear_x = base_command.get('linear_x', 0.0)
        
        # Check if robot is moving forward (positive linear_x > threshold)
        if linear_x > 0.05:  # Forward movement threshold
            if not self.is_currently_moving_forward:
                # Just started moving forward
                self.forward_start_time = current_time
                self.is_currently_moving_forward = True
                self.get_logger().info(f"🔄 Started forward movement - tracking time limit of {self.FORWARD_TIME_LIMIT}s")
            
            # Add to total forward time
            time_since_last_check = current_time - self.last_movement_check_time
            self.total_forward_time += min(time_since_last_check, duration)  # Don't over-count
            
            # Check if we've exceeded the 3-second limit
            if self.total_forward_time >= self.FORWARD_TIME_LIMIT:
                self.get_logger().error(f"🚨 FORWARD TIME LIMIT EXCEEDED! {self.total_forward_time:.1f}s >= {self.FORWARD_TIME_LIMIT}s")
                self.get_logger().error(f"🔄 FORCING BACKWARD MOVEMENT - NO EXCEPTIONS!")
                return True  # Signal that backward movement is REQUIRED
                
        else:
            # Not moving forward - reset tracking
            if self.is_currently_moving_forward:
                self.get_logger().info(f"✅ Forward movement stopped after {self.total_forward_time:.1f}s")
                self.reset_forward_tracking()
        
        self.last_movement_check_time = current_time
        return False  # No forced backward needed
    
    def reset_forward_tracking(self):
        """Reset forward movement tracking"""
        self.forward_start_time = None
        self.total_forward_time = 0.0
        self.is_currently_moving_forward = False
        self.get_logger().info("🔄 Forward movement tracking reset")
    
    def create_forced_backward_movement(self, duration=1.0):
        """Create FORCED backward movement to counter forward time limit"""
        backward_speed = 0.4  # Strong backward speed
        
        self.get_logger().error(f"🚨 CREATING FORCED BACKWARD MOVEMENT: {backward_speed} m/s for {duration}s")
        
        return {
            'movement_type': 'forced_backward_enforcement',
            'category': 'base',
            'servo_positions': {},  # No servo movement during forced backward
            'base_command': {
                'linear_x': -backward_speed,  # NEGATIVE = BACKWARD
                'linear_y': 0.0,
                'angular_z': 0.0
            }
        }

    def start_keyboard_monitoring(self):
        """Start monitoring keyboard for 'S' key press"""
        try:
            self.original_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            self.keyboard_monitoring = True
            self.get_logger().info("🎹 Keyboard monitoring started - Press 'S' to stop robot!")
        except Exception as e:
            self.get_logger().warn(f"Could not start keyboard monitoring: {e}")

    def stop_keyboard_monitoring(self):
        """Stop keyboard monitoring and restore terminal"""
        if self.original_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)
                self.keyboard_monitoring = False
                self.get_logger().info("🎹 Keyboard monitoring stopped")
            except Exception as e:
                self.get_logger().warn(f"Could not restore terminal: {e}")

    def check_keyboard_input(self):
        """Check for keyboard input without blocking"""
        if not self.keyboard_monitoring:
            return None
            
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                return key
        except Exception:
            pass
        return None

    def keyboard_monitor_thread(self):
        """Thread function to monitor keyboard input"""
        while self.performance_active and not self.emergency_stop_requested:
            key = self.check_keyboard_input()
            if key == 's':
                self.get_logger().error("🛑 'S' KEY PRESSED - STOPPING ROBOT!")
                print("\n🛑 'S' KEY DETECTED - EMERGENCY STOPPING ROBOT!")
                self.emergency_stop_requested = True
                self.performance_active = False
                
                # Send immediate stop commands
                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.linear.y = 0.0
                stop_twist.angular.z = 0.0
                
                for i in range(5):
                    print(f"🛑 S-key stop command {i+1}/5 to ALL robots")
                    self.publish_wheel_command_to_all_robots(stop_twist)
                    time.sleep(1.0)
                
                self.emergency_stop()
                break
            elif key:
                self.get_logger().info(f"Key pressed: {key} (Press 'S' to stop)")
            
            time.sleep(0.25)  # Check every 250ms - reduced to prevent audio interference

    def start_performance(self):
        """Start the choreographed performance with robust execution"""
        self.get_logger().info("Starting performance preparation...")
        
        # Wait for servo states
        t0 = time.time()
        while not self.pose_received and time.time() - t0 < 10.0:
            time.sleep(0.1)
            
        if not self.pose_received:
            self.get_logger().warn("No servo_states received; using default home positions.")
            for sid in self.servo_ids:
                self.current_pose[sid] = 500
                
        # Update home positions from current pose
        for sid in self.servo_ids:
            self.home_positions[sid] = self.current_pose.get(sid, 500)
            
        self.get_logger().info(f"Home positions: {self.home_positions}")
        
        # Start keyboard monitoring for 'S' key
        self.start_keyboard_monitoring()
        
        # Reset forward movement tracking for fresh start
        self.reset_forward_tracking()
        
        # Start the buffered execution
        self.execute_with_buffer()

    def execute_with_buffer(self):
        """Execute choreography with buffer system to prevent mid-performance stops"""
        self.performance_active = True
        self.emergency_stop_requested = False
        
        self.get_logger().info(f"Starting buffered execution with {self.buffer_time}s buffer...")
        
        # Pre-load all movements into a buffer
        movement_buffer = []
        for movement in self.choreography_timeline:
            # Create both servo and base messages
            servo_msg, base_msg = self.create_movement_messages(movement)
            
            movement_buffer.append({
                'timestamp': movement['start_time'],
                'servo_msg': servo_msg,
                'base_msg': base_msg,
                'movement_type': movement['movement_type'],
                'movement_category': movement['movement_commands']['category'],
                'duration': movement['duration'],
                'features': movement.get('features', {})  # Include segment features for robot_3
            })
        
        # Start audio with high priority and buffer delay
        audio_thread = threading.Thread(target=self.play_audio_delayed, daemon=True)
        audio_thread.start()
        
        # Start execution with precise timing (main dance thread)
        execution_thread = threading.Thread(target=self.precise_execution, args=(movement_buffer,), daemon=True)
        execution_thread.start()
        
        # Combine safety and keyboard monitoring into single thread to reduce overhead
        monitor_thread = threading.Thread(target=self.combined_monitor_thread, daemon=True)
        monitor_thread.start()
        
        self.get_logger().info("Performance started with bulletproof timing system!")
        self.get_logger().info("🎹 Press 'S' key anytime to stop the robot!")

    def combined_monitor_thread(self):
        """Combined safety and keyboard monitoring to reduce thread overhead"""
        # Calculate safety timeout
        max_time = self.song_duration + self.buffer_time + 3.0
        start_time = time.time()
        
        self.get_logger().info(f"🛡️ Combined monitor active - safety timeout: {max_time:.1f}s")
        
        while self.performance_active and not self.emergency_stop_requested:
            # Check safety timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time:
                self.get_logger().warn("⏰ SAFETY TIMEOUT - Force stopping dance!")
                self.emergency_stop_requested = True
                self.performance_active = False
                # Force return to home
                time.sleep(0.5)
                self.return_to_home(emergency=True)
                break
            
            # Check keyboard input (less frequently to reduce audio interference)
            key = self.check_keyboard_input()
            if key == 's':
                self.get_logger().error("🛑 'S' KEY PRESSED - STOPPING ROBOT!")
                print("\n🛑 'S' KEY DETECTED - EMERGENCY STOPPING ROBOT!")
                self.emergency_stop_requested = True
                self.performance_active = False
                
                # Send immediate stop commands
                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.linear.y = 0.0
                stop_twist.angular.z = 0.0
                
                for i in range(5):
                    print(f"🛑 S-key stop command {i+1}/5 to ALL robots")
                    self.publish_wheel_command_to_all_robots(stop_twist)
                    time.sleep(1.0)
                
                self.emergency_stop()
                break
            elif key:
                self.get_logger().info(f"Key pressed: {key} (Press 'S' to stop)")
            
            # Sleep longer to reduce CPU usage and audio interference
            time.sleep(0.5)  # Check every 500ms instead of multiple threads at different rates

    def safety_monitor(self):
        """Safety monitor that forces stop after song duration"""
        # Wait for song duration + buffer + safety margin
        max_time = self.song_duration + self.buffer_time + 3.0
        
        self.get_logger().info(f"🛡️ Safety monitor active - will force stop after {max_time:.1f}s")
        
        time.sleep(max_time)
        
        if self.performance_active:
            self.get_logger().warn("⏰ SAFETY TIMEOUT - Force stopping dance!")
            self.emergency_stop_requested = True
            self.performance_active = False
            
            # Force return to home
            time.sleep(0.5)
            self.return_to_home(emergency=True)

    def play_audio_delayed(self):
        """Play audio with buffer delay and monitor completion"""
        time.sleep(self.buffer_time)  # Wait for buffer period
        self.play_audio()
        
        # Audio finished - force stop everything immediately
        self.get_logger().info("🎵 Audio completed - forcing immediate stop!")
        self.emergency_stop_requested = True
        self.performance_active = False

    def precise_execution(self, movement_buffer):
        """Execute movements with microsecond precision"""
        # Wait for buffer period to build up movement queue
        time.sleep(self.buffer_time)
        
        # Get precise start time using performance counter
        start_time = time.perf_counter()
        
        # Calculate maximum safe execution time (song duration + buffer + safety margin)
        max_execution_time = self.song_duration + self.buffer_time + 5.0
        
        self.get_logger().info(f"Precise execution started! Max time: {max_execution_time:.1f}s")
        
        for buffered_movement in movement_buffer:
            # Check for stop conditions
            current_time = time.perf_counter() - start_time
            
            if self.emergency_stop_requested:
                self.get_logger().info("🛑 Emergency stop detected, halting execution")
                break
            
            if current_time > max_execution_time:
                self.get_logger().warn(f"⏰ Safety timeout reached ({max_execution_time:.1f}s) - stopping dance")
                self.emergency_stop_requested = True
                break
            
            if not self.performance_active:
                self.get_logger().info("🎵 Performance deactivated - stopping execution")
                break
                
            # Calculate precise target time
            target_time = start_time + buffered_movement['timestamp']
            current_time = time.perf_counter()
            
            # Wait with high precision
            sleep_time = target_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Final check before executing movement
            if self.emergency_stop_requested or not self.performance_active:
                self.get_logger().info("🚫 Stop condition detected - skipping movement")
                break
            
            # Execute movement (pre-calculated, no processing delay)
            # Always publish servo message to ALL robots (may contain complementary movements)
            if buffered_movement['servo_msg']:
                self.servo_pub_robot1.publish(buffered_movement['servo_msg'])
                self.servo_pub_robot2.publish(buffered_movement['servo_msg'])
                self.servo_pub_robot3.publish(buffered_movement['servo_msg'])
                # self.get_logger().debug("📡 Sent servo command to all robots")  # Disabled for audio performance
            
            # PURE SPINNING DANCE - Always send wheel commands for synchronized arms + spins
            if buffered_movement['base_msg']:
                # NO TRACKING NEEDED - PURE SPINNING HAS NO DISPLACEMENT!
                # Send to ALL robots (robot_1&2: spins, robot_3: amplitude-based linear)
                self.publish_wheel_command_to_all_robots(buffered_movement['base_msg'], buffered_movement.get('features', {}))
                movement_category = buffered_movement['movement_category']
                # Disable debug logging during performance to prevent audio stuttering
                # if movement_category == 'base':
                #     self.get_logger().debug("🌀 Sent spinning commands to robot_1&2, linear commands to robot_3")
                # else:
                #     self.get_logger().debug("🤖 Sent complementary movements to all robots (different per robot)")
                pass  # No debug logging during performance
            
            # DISABLE LOGGING DURING PERFORMANCE - prevents audio stuttering
            # Log execution only for debugging (disabled for smooth audio playback)
            # if len(movement_buffer) < 50:  # Only log for very short performances
            #     category = buffered_movement['movement_category']
            #     movement_type = buffered_movement['movement_type']
            #     self.get_logger().info(f"Executed: {movement_type} ({category})")
            pass  # No logging during performance to prevent audio stuttering
            
            # Check for stop after each movement
            if self.emergency_stop_requested or not self.performance_active:
                self.get_logger().info("🛑 Stop detected after movement - breaking loop")
                break
        
        # Performance complete - ALWAYS return to home
        self.get_logger().info("Performance complete! Returning to home position.")
        
        # Force stop all movement first
        self.stop_all_movement()
        
        # ADDITIONAL WHEEL STOP after performance - be extra sure
        self.get_logger().info("🛑 Extra wheel stop after performance completion")
        
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.linear.y = 0.0
        stop_twist.angular.z = 0.0
        
        for i in range(3):
            self.get_logger().info(f"🛑 Performance end stop command {i+1}/3 to ALL robots")
            self.publish_wheel_command_to_all_robots(stop_twist)
            time.sleep(1.0)
        
        # Wait a moment then return to home
        time.sleep(0.5)
        self.return_to_home()
        
        # Wait to ensure home position is reached
        time.sleep(1.0)
        
        # Send home command again to be absolutely sure
        self.return_to_home()
        
        # FINAL wheel stop command
        for i in range(3):
            self.get_logger().info(f"🛑 Final wheel stop {i+1}/3 to ALL robots")
            self.publish_wheel_command_to_all_robots(stop_twist)
            time.sleep(1.0)
        
        self.performance_active = False
        
        # Stop keyboard monitoring
        self.stop_keyboard_monitoring()
        
        self.get_logger().info("✅ Triple robot hybrid dance complete - All arms returned to home, all wheels STOPPED")

    def create_movement_messages(self, movement):
        """Pre-create both servo and base messages to eliminate runtime processing"""
        movement_commands = movement['movement_commands']
        
        # Create servo message
        servo_msg = None
        if movement_commands['servo_positions']:
            servo_msg = ServosPosition()
            servo_msg.position_unit = 'pulse'
            servo_msg.duration = float(movement['duration'])
            
            for sid, position in movement_commands['servo_positions'].items():
                servo_pos = ServoPosition()
                servo_pos.id = sid
                servo_pos.position = float(position)
                servo_msg.position.append(servo_pos)
        
        # Create base movement message with PURE SPINNING safety check
        base_msg = None
        base_command = movement_commands['base_command']
        
        # 🚨 FINAL SAFETY CHECK: FORCE ZERO LINEAR MOVEMENT! 🚨
        # This is our last line of defense for 50cm constraint space
        safe_linear_x = 0.0  # ALWAYS ZERO - NO EXCEPTIONS!
        safe_linear_y = 0.0  # ALWAYS ZERO - NO EXCEPTIONS!
        safe_angular_z = float(base_command['angular_z'])  # Only angular allowed
        
        # Only create message if there's actual rotation
        if abs(safe_angular_z) > 0.01:  # Threshold for rotational movements only
            base_msg = Twist()
            base_msg.linear.x = safe_linear_x   # GUARANTEED ZERO
            base_msg.linear.y = safe_linear_y   # GUARANTEED ZERO  
            base_msg.angular.z = safe_angular_z # Only rotation allowed
            
            # SUPER QUICK BURSTS: Force all movements to be very short for constrained space
            movement_duration = movement['duration']
            if movement_duration > 0.1:  # Maximum 0.1 seconds for ANY movement - SUPER QUICK!
                # For longer musical segments, we'll just execute the same fast move for the shorter time
                # This creates dynamic, punchy movements that stay in place
                pass  # Keep full speed but movement will be executed for shorter time by the choreography system
            
        return servo_msg, base_msg

    def publish_wheel_command_to_all_robots(self, twist_msg, segment_features=None):
        """Helper function to publish wheel commands to robot_1, robot_2, and robot_3"""
        # Robot_1 and Robot_2 use the standard spin-based movement
        self.cmd_vel_pub_robot1.publish(twist_msg)
        self.cmd_vel_pub_robot2.publish(twist_msg)
        
        # Robot_3 uses different movement logic (amplitude-based linear movement)
        robot3_twist = self.calculate_robot3_movement(twist_msg, segment_features)
        self.cmd_vel_pub_robot3.publish(robot3_twist)
    
    def calculate_robot3_movement(self, original_twist, segment_features):
        """Calculate robot_3 specific movement - amplitude-based linear movement"""
        robot3_twist = Twist()
        
        # Robot_3 uses amplitude-based linear movement instead of spins
        if segment_features is not None:
            # Increment beat counter for robot_3's timing pattern
            self.beat_counter += 1
            
            # Calculate amplitude from musical energy (similar to sample code)
            energy = segment_features.get('energy', 1.0)
            brightness = segment_features.get('brightness', 1.0)
            
            # Amplitude calculation (similar to sample: amp = (loud / median_loud) * max_delta)
            amp = energy * brightness * 200.0  # Scale factor for movement intensity
            
            # Robot_3 moves every 8th beat when amplitude is high enough (like sample code)
            if self.beat_counter % 8 == 0 and amp > 200:
                # Alternate direction (like sample: self.last_move_direction *= -1)
                self.robot3_move_direction *= -1
                
                # Linear movement based on amplitude (like sample: 0.001 * amp * direction)
                robot3_twist.linear.x = 0.001 * amp * self.robot3_move_direction
                robot3_twist.linear.y = 0.0
                robot3_twist.angular.z = 0.0
                
                self.get_logger().debug(f"Robot_3 linear movement: {robot3_twist.linear.x:.3f} (amp={amp:.1f}, beat={self.beat_counter})")
            else:
                # No movement (like sample: publish Twist() to stop)
                robot3_twist.linear.x = 0.0
                robot3_twist.linear.y = 0.0
                robot3_twist.angular.z = 0.0
        else:
            # If no segment features, use original twist for robot_3 (fallback)
            robot3_twist = original_twist
        
        return robot3_twist

    def stop_all_movement(self):
        """Stop all servo and base movements immediately"""
        # FORCE STOP BASE MOVEMENTS - send multiple stop commands rapidly
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.linear.y = 0.0
        stop_twist.angular.z = 0.0
        
        # Send stop commands slowly and repeatedly - 1 per second to BOTH ROBOTS
        for i in range(5):  # 5 stop commands at 1 second intervals
            self.get_logger().info(f"🛑 Sending stop command {i+1}/5 to ALL robots")
            self.publish_wheel_command_to_all_robots(stop_twist)
            time.sleep(1.0)  # 1 second between commands
        
        # FORCE STOP ALL SERVOS - send immediate home command
        self.force_servo_stop()
        
        self.get_logger().info("All movements stopped - wheels and servos")

    def force_servo_stop(self):
        """FORCE all servos to stop immediately - multiple attempts - ALL ROBOTS"""
        self.get_logger().warn("🛑 FORCING SERVO STOP ON ALL ROBOTS!")
        
        # Send stop command 10 times rapidly to all robots
        for attempt in range(10):
            stop_msg = ServosPosition()
            stop_msg.position_unit = 'pulse'
            stop_msg.duration = 0.1  # Very fast movement to stop
            
            for sid in self.servo_ids:
                servo_pos = ServoPosition()
                servo_pos.id = sid
                servo_pos.position = 500.0  # Force to center
                stop_msg.position.append(servo_pos)
            
            # Send to all robots simultaneously
            self.servo_pub_robot1.publish(stop_msg)
            self.servo_pub_robot2.publish(stop_msg)
            self.servo_pub_robot3.publish(stop_msg)
            time.sleep(0.05)
        
        self.get_logger().warn(f"🛑 Sent {10} FORCE STOP commands to ALL ROBOTS!")

    def emergency_stop(self):
        """Instant emergency stop - halts everything immediately"""
        self.get_logger().error("🚨🚨🚨 EMERGENCY STOP ACTIVATED! 🚨🚨🚨")
        
        # Set emergency flag FIRST - multiple times
        for _ in range(5):
            self.emergency_stop_requested = True
            self.performance_active = False
        
        # Kill audio process HARD
        if self.audio_process:
            try:
                self.audio_process.kill()
                self.audio_process.wait(timeout=0.1)
            except:
                pass
        
        # FORCE STOP EVERYTHING - multiple rapid attempts
        for emergency_attempt in range(5):  # More attempts
            self.get_logger().error(f"🚨 Emergency attempt {emergency_attempt + 1}/5")
            
            # AGGRESSIVE WHEEL STOPPING - more stop commands
            stop_twist = Twist()
            stop_twist.linear.x = 0.0
            stop_twist.linear.y = 0.0
            stop_twist.angular.z = 0.0
            
            for j in range(3):  # 3 stop commands per emergency attempt
                self.get_logger().error(f"🛑 Emergency stop command {j+1}/3 in attempt {emergency_attempt + 1} to ALL robots")
                self.publish_wheel_command_to_all_robots(stop_twist)
                time.sleep(1.0)  # 1 second between commands
            
            # Force stop servos immediately
            self.force_servo_stop()
            
            # Multiple home commands
            self.return_to_home(emergency=True)
            time.sleep(0.1)  # Shorter wait between attempts
        
        # Final home command
        self.return_to_home(emergency=True)
        
        # FINAL AGGRESSIVE WHEEL STOP - make absolutely sure wheels stop
        self.get_logger().error("🛑 FINAL WHEEL STOP - MAKING ABSOLUTELY SURE!")
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.linear.y = 0.0  
        stop_twist.angular.z = 0.0
        
        for i in range(5):  # 5 final stop commands
            self.get_logger().error(f"🛑 Final stop command {i+1}/5 to ALL robots")
            self.publish_wheel_command_to_all_robots(stop_twist)
            time.sleep(1.0)  # 1 second between commands
        
        # Publish emergency stop signal
        emergency_msg = Bool()
        emergency_msg.data = True
        for _ in range(5):
            self.emergency_stop_pub.publish(emergency_msg)
            time.sleep(0.02)
        
        # Stop keyboard monitoring
        self.stop_keyboard_monitoring()
        
        # Reset forward movement tracking
        self.reset_forward_tracking()
        
        self.get_logger().error("🚨 EMERGENCY STOP COMPLETE - ALL SYSTEMS HALTED - WHEELS FORCED STOP")

    def return_to_home(self, emergency=False):
        """Return all servos to home position - GUARANTEED - ALL ROBOTS"""
        duration = 0.5 if emergency else 2.0
        
        # Create home position command with default 500 position
        home_msg = ServosPosition()
        home_msg.position_unit = 'pulse'
        home_msg.duration = duration
        
        for sid in self.servo_ids:
            servo_pos = ServoPosition()
            servo_pos.id = sid
            # Always use 500 as default home position for reliability
            servo_pos.position = 500.0
            home_msg.position.append(servo_pos)
        
        # Send home command multiple times to all robots to ensure it's received
        for i in range(3):
            self.servo_pub_robot1.publish(home_msg)
            self.servo_pub_robot2.publish(home_msg)
            self.servo_pub_robot3.publish(home_msg)
            time.sleep(0.1)
        
        self.get_logger().info(f"🏠 Sent home command to ALL ROBOTS - All servos to position 500")

    def play_audio(self):
        """Play audio with optimized process management and buffering"""
        # Optimized command configurations for different players
        if self.audio_player == 'mpg123':
            # Optimized mpg123 for low latency and stable playback
            cmd = [self.audio_player, '-q', '--buffer', '1024', '--preload', '0.2', self.audio_path]
        elif self.audio_player == 'ffplay':
            # Optimized ffplay with better buffering and no video display
            cmd = [self.audio_player, '-nodisp', '-autoexit', '-sync', 'audio', 
                   '-framedrop', '-infbuf', self.audio_path]
        elif self.audio_player == 'aplay':
            # ALSA player - often most stable on Linux
            cmd = [self.audio_player, '-q', self.audio_path]
        elif self.audio_player == 'paplay':
            # PulseAudio player - good for Ubuntu systems
            cmd = [self.audio_player, '--verbose=0', self.audio_path]
        else:
            # Fallback for other players
            cmd = [self.audio_player, self.audio_path]
        
        def safe_nice():
            """Safely adjust process priority without failing if permissions are insufficient"""
            try:
                import os
                # Try to increase priority, but don't fail if we can't
                os.nice(-5)  # Higher priority for audio
            except (OSError, PermissionError):
                # If we can't increase priority, just continue normally
                pass
        
        try:
            # Use higher process priority for audio to prevent stuttering
            self.audio_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                preexec_fn=safe_nice  # Safe priority adjustment
            )
            self.audio_process.wait()
        except Exception as e:
            self.get_logger().error(f"Audio playback error: {e}")
            # Try fallback audio command without any priority adjustment
            try:
                fallback_cmd = [self.audio_player, self.audio_path]
                self.audio_process = subprocess.Popen(
                    fallback_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                    # No preexec_fn in fallback to avoid any issues
                )
                self.audio_process.wait()
            except Exception as e2:
                self.get_logger().error(f"Fallback audio playback also failed: {e2}")
        finally:
            self.audio_process = None

def main():
    parser = argparse.ArgumentParser(description="Advanced JetRover AI Choreography Engine")
    parser.add_argument('--audio', default=default_audio, help="Path to audio file")
    parser.add_argument('--player', default='mpg123', choices=['mpg123', 'ffplay', 'aplay', 'paplay', 'mpg321', 'mplayer'], help="Audio player (mpg123 recommended for best performance)")
    parser.add_argument('--buffer', type=float, default=2.0, help="Buffer time to prevent performance interruptions")
    parser.add_argument('--energy', type=float, default=0.85, help="Energy scale factor (0.1-1.0): lower = gentler movements")
    parser.add_argument('--start', action='store_true', help="Start performance immediately")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = None
    
    try:
        # Create the advanced dance node
        node = AdvancedDanceNode(args.audio, args.player, args.buffer, args.energy)
        
        if args.start:
            # Start performance automatically
            node.get_logger().info("Auto-starting performance...")
            performance_thread = threading.Thread(target=node.start_performance, daemon=True)
            performance_thread.start()
        else:
            # Wait for manual trigger
            node.get_logger().info("Node ready! Send 'True' to /dance/start_command or call start_performance() to begin")
            
            # Add start command subscriber for manual triggering
            def start_command_cb(msg):
                if msg.data and not node.performance_active:
                    node.get_logger().info("Start command received!")
                    performance_thread = threading.Thread(target=node.start_performance, daemon=True)
                    performance_thread.start()
            
            node.create_subscription(Bool, '/dance/start_command', start_command_cb, 10)
        
        # Keep node running
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\n🚨 CTRL+C DETECTED - EMERGENCY STOPPING ROBOT!")
        if node:
            node.get_logger().error("🚨 KEYBOARD INTERRUPT - FORCING IMMEDIATE STOP!")
            # Force stop everything immediately
            node.emergency_stop_requested = True
            node.performance_active = False
            
            # SLOW STOP COMMANDS for Ctrl+C - 1 per second
            if hasattr(node, 'cmd_vel_pub'):
                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.linear.y = 0.0
                stop_twist.angular.z = 0.0
                print("🛑 SENDING STOP COMMANDS - 1 PER SECOND...")
                for i in range(5):
                    print(f"🛑 Stop command {i+1}/5 to ALL robots")
                    node.publish_wheel_command_to_all_robots(stop_twist)
                    time.sleep(1.0)  # 1 second delay between commands
                print("🛑 STOP COMMANDS COMPLETED!")
            
            if hasattr(node, 'emergency_stop'):
                node.emergency_stop()
            
            print("✅ ROBOT SHOULD BE STOPPED!")
    except Exception as e:
        if node:
            node.get_logger().error(f"Unexpected error: {e}")
            if hasattr(node, 'emergency_stop'):
                node.emergency_stop()
        else:
            print(f"Error during initialization: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
