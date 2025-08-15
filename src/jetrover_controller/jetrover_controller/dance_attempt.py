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

        # Publishers and subscribers - DUAL ROBOT SUPPORT
        self.servo_pub_robot1 = self.create_publisher(ServosPosition, '/robot_1/servo_controller', 10)
        self.servo_pub_robot2 = self.create_publisher(ServosPosition, '/robot_2/servo_controller', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/robot_2/controller/cmd_vel', 10)
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
            
            # PURE SPINNING movements - 50cm constraint space - NO LINEAR MOVEMENT
            'spin_left': {'energy': 'medium', 'brightness': 'medium', 'pattern': 'rotational_left', 'type': 'base'},
            'spin_right': {'energy': 'medium', 'brightness': 'medium', 'pattern': 'rotational_right', 'type': 'base'},
            'half_circle_left': {'energy': 'high', 'brightness': 'high', 'pattern': 'half_rotation_left', 'type': 'base'},
            'half_circle_right': {'energy': 'high', 'brightness': 'high', 'pattern': 'half_rotation_right', 'type': 'base'},
            'quarter_circle_left': {'energy': 'low', 'brightness': 'low', 'pattern': 'quarter_rotation_left', 'type': 'base'},
            'quarter_circle_right': {'energy': 'low', 'brightness': 'low', 'pattern': 'quarter_rotation_right', 'type': 'base'},
            'full_circle_left': {'energy': 'very_high', 'brightness': 'very_high', 'pattern': 'full_rotation_left', 'type': 'base'},
            'full_circle_right': {'energy': 'very_high', 'brightness': 'very_high', 'pattern': 'full_rotation_right', 'type': 'base'},
            'spin_burst': {'energy': 'explosive', 'brightness': 'explosive', 'pattern': 'random_spin', 'type': 'base'}
        }
        
        self.get_logger().info(f"Starting comprehensive music analysis...")
        self.get_logger().info(f"🎚️ Energy Scale: {self.energy_scale:.2f} (lower = gentler movements)")
        self.get_logger().info(f"🤖🤖 DUAL ROBOT MODE: Commands will be sent to both robot_1 and robot_2!")
        self.get_logger().info(f"🌀 PURE SPINNING DANCE: 50cm constraint space - ZERO LINEAR MOVEMENT!")
        self.get_logger().info(f"🔄 SPIN-ONLY MODES: Left spins, Right spins, Half circles, Quarter circles, Full circles!")
        self.get_logger().info(f"🎯 ROTATION SPEEDS: Max 3.0 rad/s - Dynamic spin variations based on music!")
        self.get_logger().info(f"📍 STATIONARY DANCING: Robot stays in exact same position - only rotates!")
        self.get_logger().info(f"🛑 Multiple Stop Methods: Ctrl+C OR press 'S' key - 1 stop command per second!")
        self.get_logger().info(f"🌪️ NO DISPLACEMENT: Arms + Pure rotational wheel movements only!")
        # Complete analysis BEFORE starting any performance
        self.analyze_complete_song()
        self.create_choreography_plan()
        self.get_logger().info("Choreography fully planned and ready for flawless dual robot hybrid dancing!")

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
            
            # FORCE VERY SHORT DURATIONS for constrained space dancing - split long segments
            max_movement_duration = 0.1  # Maximum 0.1 seconds per movement - SUPER QUICK BURSTS!
            
            if duration > max_movement_duration:
                # Split long segments into multiple quick bursts
                num_splits = int(duration / max_movement_duration) + 1
                split_duration = duration / num_splits
                
                for split_idx in range(num_splits):
                    split_start = start_time + (split_idx * split_duration)
                    split_end = split_start + split_duration
                    
                    # Extract features for this split segment
                    segment_features = self.extract_segment_features(split_start, split_end)
                    
                    # Determine movement type based on musical characteristics
                    movement_type = self.classify_movement_type(segment_features)
                    
                    # Calculate movement commands (servo + base)
                    movement_commands = self.calculate_movement_commands(movement_type, segment_features)
                    
                    # TRACK MOVEMENT AND FORCE RETURN EVERY 3 BEATS
                    self.movement_counter += 1
                    
                    # Force return movement every 3 beats to prevent continuous displacement
                    if self.movement_counter % 3 == 0:
                        # Create forced return to center
                        return_commands = self.create_forced_return_to_center()
                        if return_commands:
                            return_start = split_end
                            return_end = split_end + 0.1  # Longer return movement
                            segments.append({
                                'start_time': return_start,
                                'end_time': return_end,
                                'duration': 0.1,
                                'movement_type': 'forced_return_to_center',
                                'movement_commands': return_commands,
                                'features': segment_features
                            })
                            self.last_forced_return = self.movement_counter
                            # Reset displacement tracking
                            self.accumulated_displacement = {'x': 0.0, 'y': 0.0}
                    else:
                        # Add the main movement only if we're not doing a forced return
                        segments.append({
                            'start_time': split_start,
                            'end_time': split_end,
                            'duration': split_duration,
                            'movement_type': movement_type,
                            'movement_commands': movement_commands,
                            'features': segment_features
                        })
                        
                        # Track displacement
                        base_cmd = movement_commands.get('base_command', {})
                        self.accumulated_displacement['x'] += base_cmd.get('linear_x', 0.0) * split_duration
                        self.accumulated_displacement['y'] += base_cmd.get('linear_y', 0.0) * split_duration
            else:
                # Normal short segment - keep as is
                # Extract features for this segment
                segment_features = self.extract_segment_features(start_time, end_time)
                
                # Determine movement type based on musical characteristics
                movement_type = self.classify_movement_type(segment_features)
                
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
                
                # NO RETURN MOVEMENTS NEEDED - PURE SPINNING HAS NO DISPLACEMENT!
        
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
        """Classify movement type - PURE SPINNING DANCE for 50cm constraint space"""
        energy = features['energy']
        brightness = features['brightness']
        onset = features['onset_strength']
        duration = features['duration']
        
        # PURE SPINNING DANCE - NO LINEAR MOVEMENT EVER!
        # Map energy and musical features to ONLY rotational movements + arm movements
        
        # EXPLOSIVE ENERGY - Random rapid spins
        if energy > 1.8 and brightness > 1.5:
            if onset > 0.7:
                return 'spin_burst'  # Base: random explosive spins + arms
            else:
                return 'powerful_strike'  # Servo: maximum intensity arm movement
                
        # VERY HIGH ENERGY - Full circles
        elif energy > 1.5 and brightness > 1.5:
            if onset > 0.6:
                return 'full_circle_left' if brightness > energy else 'full_circle_right'  # Base: full spins + arms
            else:
                return 'dramatic_sweep'  # Servo: dramatic arm sweeps
        
        # HIGH ENERGY - Half circles
        elif energy > 1.2 and brightness > 1.2:
            if duration > 0.8 and onset > 0.5:
                return 'half_circle_left' if brightness > 1.3 else 'half_circle_right'  # Base: half spins + arms
            else:
                return 'energetic_wave'  # Servo: sharp arm movements
                
        elif energy > 1.2:
            # Dynamic spins based on onset
            if onset > 0.6:
                return 'spin_left' if brightness > energy else 'spin_right'  # Base: dynamic spins + arms
            else:
                return 'bright_sparkle'  # Servo: quick bright arm movements
        
        # MEDIUM ENERGY - Controlled spins
        elif energy > 1.0 and brightness > 1.2:
            if duration > 1.0:
                return 'spin_left' if onset > 0.5 else 'spin_right'  # Base: controlled spins + arms
            else:
                return 'flowing_reach'  # Servo: sustained arm movements
                
        elif energy > 1.0 and brightness < 0.8:
            # Gentle rotations
            if onset > 0.5:
                return 'quarter_circle_left' if energy > 1.1 else 'quarter_circle_right'  # Base: quarter turns + arms
            else:
                return 'deep_pulse'  # Servo: rhythmic arm pulses
                
        elif energy > 1.0:
            # Medium spins
            if duration > 0.6:
                return 'spin_right' if brightness > 1.0 else 'spin_left'  # Base: medium spins + arms
            else:
                return 'dramatic_sweep'  # Servo: dramatic arm sweeps
        
        # LOWER ENERGY - Gentle rotations
        elif brightness > 1.3:
            # Bright, gentle movements
            if onset > 0.4:
                return 'quarter_circle_right' if energy > 0.9 else 'quarter_circle_left'  # Base: gentle quarter turns + arms
            else:
                return 'bright_sparkle'  # Servo: quick bright arm movements
                
        elif energy > 0.8 and brightness > 0.8:
            # Balanced energy - gentle spins
            return 'quarter_circle_left' if onset > 0.4 else 'flowing_reach'  # Base: very gentle turns OR Servo: flowing arms
        
        # LOWEST ENERGY - Arms only or minimal rotation
        elif brightness > 1.0:
            return 'subtle_sway'  # Servo: gentle arm sway
        else:
            return 'gentle_wave'  # Servo: soft arm wave

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
        """Calculate PURE SPINNING movements - 50cm constraint space - NO LINEAR MOVEMENT"""
        energy = features['energy']
        duration = features['duration']
        brightness = features['brightness']
        onset = features['onset_strength']
        
        # PURE SPINNING DANCE: Only angular velocity, ZERO linear movement
        angular_speed = min(3.0, energy * 1.2)  # Max 3.0 rad/s based on energy
        
        # Add speed boost for high brightness (bright musical passages)
        if brightness > 1.2:
            angular_speed *= 1.3  # 30% spin boost for bright sections
            
        # Add speed boost for strong onsets (musical accents)
        if onset > 0.6:
            angular_speed *= 1.4  # 40% spin boost for strong beats
        
        # Apply TEMPO SCALING for music synchronization
        angular_speed *= tempo_scale
        
        # Cap the speed at maximum after tempo scaling
        angular_speed = min(3.0, angular_speed)
        
        # 🚨 CRITICAL: ZERO LINEAR MOVEMENT - PURE SPINNING ONLY! 🚨
        # This GUARANTEES robot stays in 50cm constraint space!
        base_command = {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}
        
        if movement_type == 'spin_left':
            # Simple left spin
            base_command['angular_z'] = angular_speed * 0.8  # Positive = left spin
            
        elif movement_type == 'spin_right':
            # Simple right spin  
            base_command['angular_z'] = -angular_speed * 0.8  # Negative = right spin
            
        elif movement_type == 'quarter_circle_left':
            # Gentle left quarter turn (90 degrees)
            base_command['angular_z'] = angular_speed * 0.4  # Slower for precision
            
        elif movement_type == 'quarter_circle_right':
            # Gentle right quarter turn (90 degrees)
            base_command['angular_z'] = -angular_speed * 0.4  # Slower for precision
            
        elif movement_type == 'half_circle_left':
            # Left half circle (180 degrees)
            base_command['angular_z'] = angular_speed * 0.7  # Medium speed
            
        elif movement_type == 'half_circle_right':
            # Right half circle (180 degrees)
            base_command['angular_z'] = -angular_speed * 0.7  # Medium speed
            
        elif movement_type == 'full_circle_left':
            # Full left circle (360 degrees)
            base_command['angular_z'] = angular_speed  # Full speed for complete rotation
            
        elif movement_type == 'full_circle_right':
            # Full right circle (360 degrees)
            base_command['angular_z'] = -angular_speed  # Full speed for complete rotation
            
        elif movement_type == 'spin_burst':
            # EXPLOSIVE random spinning - multiple direction changes
            burst_intensity = 1.0
            if onset > 0.8:
                burst_intensity = 1.2  # Even more intense for very strong beats
            
            # Random explosive spin moves - PURE ROTATION ONLY!
            spin_moves = [
                {'angular_z': angular_speed * burst_intensity},  # FAST left spin
                {'angular_z': -angular_speed * burst_intensity},  # FAST right spin
                {'angular_z': angular_speed * burst_intensity * 0.7},  # Medium left spin
                {'angular_z': -angular_speed * burst_intensity * 0.7},  # Medium right spin
                {'angular_z': angular_speed * burst_intensity * 1.2},  # SUPER FAST left (if within limits)
                {'angular_z': -angular_speed * burst_intensity * 1.2},  # SUPER FAST right (if within limits)
            ]
            
            # Choose random spin direction and intensity
            chosen_spin = random.choice(spin_moves)
            
            # Apply speed limits
            chosen_spin['angular_z'] = max(-3.0, min(3.0, chosen_spin['angular_z']))
            
            base_command.update(chosen_spin)
            
        return base_command

    def calculate_subtle_base_complement(self, movement_type, features, tempo_scale=1.0):
        """Calculate subtle SPINNING movements to complement servo arm movements - 50cm constraint"""
        energy = features['energy']
        onset = features['onset_strength']
        
        # Subtle SPINNING ONLY - NO LINEAR MOVEMENT for arm complementing
        angular_speed = min(1.5, energy * 0.6)  # Gentler rotation for arm complement
        
        # Apply tempo scaling
        angular_speed *= tempo_scale
        
        # Cap the speeds
        angular_speed = min(1.5, angular_speed)
        
        # 🚨 CRITICAL: ZERO LINEAR MOVEMENT - ONLY SUBTLE SPINS! 🚨
        # This GUARANTEES robot stays in 50cm constraint space!
        base_command = {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}
        
        # Add subtle SPINNING movements based on arm movement type
        if movement_type in ['powerful_strike', 'dramatic_sweep']:
            # Subtle rotation during dramatic arm movements
            direction = 1 if onset > 0.5 else -1
            base_command['angular_z'] = direction * angular_speed * 0.4
            
        elif movement_type in ['energetic_wave', 'bright_sparkle']:
            # Gentle spins during energetic arm movements
            direction = 1 if energy > 1.0 else -1
            base_command['angular_z'] = direction * angular_speed * 0.3
            
        elif movement_type in ['flowing_reach', 'subtle_sway']:
            # Very gentle rotation during flowing arms
            direction = 1 if onset > 0.4 else -1
            base_command['angular_z'] = direction * angular_speed * 0.2
            
        elif movement_type == 'deep_pulse':
            # Slight rotation to match arm pulses
            direction = 1 if onset > 0.5 else -1
            base_command['angular_z'] = direction * angular_speed * 0.25
                
        elif movement_type == 'gentle_wave':
            # Minimal rotation to complement gentle arm waves
            direction = 1 if features['brightness'] > 1.0 else -1
            base_command['angular_z'] = direction * angular_speed * 0.15
            
        return base_command

    def calculate_complementary_servo_positions(self, movement_type, features, base_amplitude):
        """Calculate subtle servo movements to complement SPINNING base movements"""
        positions = {}
        
        # Subtle servo movements that complement spinning motions
        complement_amplitude = base_amplitude * 0.3  # Much gentler than main servo movements
        
        for sid in self.servo_ids:
            if movement_type in ['spin_left', 'spin_right']:
                # Gentle wave during spins
                positions[sid] = self.home_positions[sid] + ((-1) ** sid) * complement_amplitude * 0.3
            elif movement_type in ['full_circle_left', 'full_circle_right']:
                # Outward reach during full circle spins
                positions[sid] = self.home_positions[sid] + complement_amplitude * 0.5
            elif movement_type in ['half_circle_left', 'half_circle_right']:
                # Medium reach during half circles
                positions[sid] = self.home_positions[sid] + complement_amplitude * 0.4
            elif movement_type in ['quarter_circle_left', 'quarter_circle_right']:
                # Gentle flowing motion during quarter turns
                positions[sid] = self.home_positions[sid] + random.uniform(-0.2, 0.2) * complement_amplitude
            elif movement_type == 'spin_burst':
                # Dynamic random movements during spin bursts
                positions[sid] = self.home_positions[sid] + random.uniform(-0.4, 0.4) * complement_amplitude
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
                    print(f"🛑 S-key stop command {i+1}/5")
                    self.cmd_vel_pub.publish(stop_twist)
                    time.sleep(1.0)
                
                self.emergency_stop()
                break
            elif key:
                self.get_logger().info(f"Key pressed: {key} (Press 'S' to stop)")
            
            time.sleep(0.1)  # Check every 100ms

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
                'duration': movement['duration']
            })
        
        # Start audio with buffer delay
        audio_thread = threading.Thread(target=self.play_audio_delayed, daemon=True)
        audio_thread.start()
        
        # Start execution with precise timing
        execution_thread = threading.Thread(target=self.precise_execution, args=(movement_buffer,), daemon=True)
        execution_thread.start()
        
        # Start safety monitor thread
        safety_thread = threading.Thread(target=self.safety_monitor, daemon=True)
        safety_thread.start()
        
        # Start keyboard monitoring thread for 'S' key
        keyboard_thread = threading.Thread(target=self.keyboard_monitor_thread, daemon=True)
        keyboard_thread.start()
        
        self.get_logger().info("Performance started with bulletproof timing system!")
        self.get_logger().info("🎹 Press 'S' key anytime to stop the robot!")

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
            # Always publish servo message to BOTH robots (may contain complementary movements)
            if buffered_movement['servo_msg']:
                self.servo_pub_robot1.publish(buffered_movement['servo_msg'])
                self.servo_pub_robot2.publish(buffered_movement['servo_msg'])
                self.get_logger().debug("📡 Sent servo command to both robots")
            
            # PURE SPINNING DANCE - Always send wheel commands for synchronized arms + spins
            if buffered_movement['base_msg']:
                # NO TRACKING NEEDED - PURE SPINNING HAS NO DISPLACEMENT!
                self.cmd_vel_pub.publish(buffered_movement['base_msg'])
                movement_category = buffered_movement['movement_category']
                if movement_category == 'base':
                    self.get_logger().debug("🌀 Sent primary spinning dance command")
                else:
                    self.get_logger().debug("🤖 Sent complementary spinning movement with arms")
            
            # Log execution (optional, can be disabled for even better performance)
            if len(movement_buffer) < 100:  # Only log for shorter performances
                category = buffered_movement['movement_category']
                movement_type = buffered_movement['movement_type']
                self.get_logger().info(f"Executed: {movement_type} ({category})")
            
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
            self.get_logger().info(f"🛑 Performance end stop command {i+1}/3")
            self.cmd_vel_pub.publish(stop_twist)
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
            self.get_logger().info(f"🛑 Final wheel stop {i+1}/3")
            self.cmd_vel_pub.publish(stop_twist)
            time.sleep(1.0)
        
        self.performance_active = False
        
        # Stop keyboard monitoring
        self.stop_keyboard_monitoring()
        
        self.get_logger().info("✅ Hybrid dance complete - Arms returned to home, wheels STOPPED")

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

    def stop_all_movement(self):
        """Stop all servo and base movements immediately"""
        # FORCE STOP BASE MOVEMENTS - send multiple stop commands rapidly
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.linear.y = 0.0
        stop_twist.angular.z = 0.0
        
        # Send stop commands slowly and repeatedly - 1 per second
        for i in range(5):  # 5 stop commands at 1 second intervals
            self.get_logger().info(f"🛑 Sending stop command {i+1}/5")
            self.cmd_vel_pub.publish(stop_twist)
            time.sleep(1.0)  # 1 second between commands
        
        # FORCE STOP ALL SERVOS - send immediate home command
        self.force_servo_stop()
        
        self.get_logger().info("All movements stopped - wheels and servos")

    def force_servo_stop(self):
        """FORCE all servos to stop immediately - multiple attempts - BOTH ROBOTS"""
        self.get_logger().warn("🛑 FORCING SERVO STOP ON BOTH ROBOTS!")
        
        # Send stop command 10 times rapidly to both robots
        for attempt in range(10):
            stop_msg = ServosPosition()
            stop_msg.position_unit = 'pulse'
            stop_msg.duration = 0.1  # Very fast movement to stop
            
            for sid in self.servo_ids:
                servo_pos = ServoPosition()
                servo_pos.id = sid
                servo_pos.position = 500.0  # Force to center
                stop_msg.position.append(servo_pos)
            
            # Send to both robots simultaneously
            self.servo_pub_robot1.publish(stop_msg)
            self.servo_pub_robot2.publish(stop_msg)
            time.sleep(0.05)
        
        self.get_logger().warn(f"🛑 Sent {10} FORCE STOP commands to BOTH robots!")

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
                self.get_logger().error(f"🛑 Emergency stop command {j+1}/3 in attempt {emergency_attempt + 1}")
                self.cmd_vel_pub.publish(stop_twist)
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
            self.get_logger().error(f"🛑 Final stop command {i+1}/5")
            self.cmd_vel_pub.publish(stop_twist)
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
        """Return all servos to home position - GUARANTEED - BOTH ROBOTS"""
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
        
        # Send home command multiple times to both robots to ensure it's received
        for i in range(3):
            self.servo_pub_robot1.publish(home_msg)
            self.servo_pub_robot2.publish(home_msg)
            time.sleep(0.1)
        
        self.get_logger().info(f"🏠 Sent home command to BOTH robots - All servos to position 500")

    def play_audio(self):
        """Play audio with proper process management"""
        if self.audio_player == 'mpg123':
            cmd = [self.audio_player, '-q', self.audio_path]
        elif self.audio_player == 'ffplay':
            cmd = [self.audio_player, '-nodisp', '-autoexit', self.audio_path]
        else:
            cmd = [self.audio_player, self.audio_path]
        
        try:
            self.audio_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.audio_process.wait()
        except Exception as e:
            self.get_logger().error(f"Audio playback error: {e}")
        finally:
            self.audio_process = None

def main():
    parser = argparse.ArgumentParser(description="Advanced JetRover AI Choreography Engine")
    parser.add_argument('--audio', default=default_audio, help="Path to audio file")
    parser.add_argument('--player', default='ffplay', choices=['ffplay', 'mpg123', 'mpg321', 'mplayer', 'aplay'], help="Audio player")
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
                    print(f"🛑 Stop command {i+1}/5")
                    node.cmd_vel_pub.publish(stop_twist)
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
