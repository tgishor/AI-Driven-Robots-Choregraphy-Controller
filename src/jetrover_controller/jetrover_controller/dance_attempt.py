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
            
            # Mecanum base movements
            'sideways_slide': {'energy': 'medium', 'brightness': 'medium', 'pattern': 'smooth', 'type': 'base'},
            'diagonal_drift': {'energy': 'medium', 'brightness': 'high', 'pattern': 'flowing', 'type': 'base'},
            'spin_in_place': {'energy': 'high', 'brightness': 'high', 'pattern': 'rotational', 'type': 'base'},
            'circular_flow': {'energy': 'medium', 'brightness': 'medium', 'pattern': 'circular', 'type': 'base'},
            'zigzag_dance': {'energy': 'high', 'brightness': 'high', 'pattern': 'sharp', 'type': 'base'},
            'smooth_glide': {'energy': 'low', 'brightness': 'low', 'pattern': 'sustained', 'type': 'base'},
            'rhythmic_steps': {'energy': 'medium', 'brightness': 'low', 'pattern': 'rhythmic', 'type': 'base'},
            'explosive_burst': {'energy': 'very_high', 'brightness': 'very_high', 'pattern': 'sudden', 'type': 'base'}
        }
        
        self.get_logger().info(f"Starting comprehensive music analysis...")
        self.get_logger().info(f"🎚️ Energy Scale: {self.energy_scale:.2f} (lower = gentler movements)")
        self.get_logger().info(f"🤖🤖 DUAL ROBOT MODE: Commands will be sent to both robot_1 and robot_2!")
        self.get_logger().info(f"🕺💃 CONSTRAINED SPACE DANCING: Arms + Wheels synchronized, stays in place!")
        self.get_logger().info(f"🚀 Speed Limits: Max 1.5 m/s linear (with auto-return), 6.0 rad/s angular for spins!")
        self.get_logger().info(f"⚡ Super Quick Bursts: Max 0.1s duration + 0.05s auto-return")
        self.get_logger().info(f"🔄 Auto-Return System: Forward movements automatically return to stay in place!")
        self.get_logger().info(f"🎲 Random Cool Moves: Fast spins, quick dashes with return, and combo moves!")
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
                    
                    # Add the main movement
                    segments.append({
                        'start_time': split_start,
                        'end_time': split_end,
                        'duration': split_duration,
                        'movement_type': movement_type,
                        'movement_commands': movement_commands,
                        'features': segment_features
                    })
                    
                    # ADD AUTOMATIC RETURN MOVEMENT for linear movements to stay in place
                    if self.has_linear_movement(movement_commands):
                        return_commands = self.create_return_movement(movement_commands, segment_features)
                        if return_commands:
                            return_start = split_end
                            return_end = split_end + 0.05  # Very quick 0.05s return movement
                            segments.append({
                                'start_time': return_start,
                                'end_time': return_end,
                                'duration': 0.05,
                                'movement_type': f"return_{movement_type}",
                                'movement_commands': return_commands,
                                'features': segment_features
                            })
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
                
                # ADD AUTOMATIC RETURN MOVEMENT for linear movements to stay in place
                if self.has_linear_movement(movement_commands):
                    return_commands = self.create_return_movement(movement_commands, segment_features)
                    if return_commands:
                        return_start = end_time
                        return_end = end_time + 0.05  # Very quick 0.05s return movement
                        segments.append({
                            'start_time': return_start,
                            'end_time': return_end,
                            'duration': 0.05,
                            'movement_type': f"return_{movement_type}",
                            'movement_commands': return_commands,
                            'features': segment_features
                        })
        
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
        """Classify movement type - ARMS + WHEELS for synchronized dancing"""
        energy = features['energy']
        brightness = features['brightness']
        onset = features['onset_strength']
        duration = features['duration']
        
        # NEW: ARMS + WHEELS SYNCHRONIZED DANCING
        # Map energy and musical features to both servo and base movements
        
        # Very high energy - combine powerful arm movements with wheel spins
        if energy > 1.8 and brightness > 1.5:
            # Alternate between powerful arms and full spins based on onset strength
            if onset > 0.7:
                return 'explosive_burst'  # Base: quick directional burst + arms
            else:
                return 'powerful_strike'  # Servo: maximum intensity arm movement
                
        elif energy > 1.5 and brightness > 1.5:
            # Combine dramatic arms with rotational movements
            if onset > 0.6:
                return 'spin_in_place'  # Base: spinning dance + arms
            else:
                return 'dramatic_sweep'  # Servo: dramatic arm sweeps
        
        # High energy movements - mix of arms and wheels
        elif energy > 1.2 and brightness > 1.2:
            # Alternate between energetic arms and lateral dancing
            if duration > 0.8 and onset > 0.5:
                return 'zigzag_dance'  # Base: quick directional changes + arms
            else:
                return 'energetic_wave'  # Servo: sharp arm movements
                
        elif energy > 1.2:
            # Quick movements - arms or subtle wheel movements
            if onset > 0.6:
                return 'rhythmic_steps'  # Base: rhythmic forward/backward + arms
            else:
                return 'bright_sparkle'  # Servo: quick bright arm movements
        
        # Medium-high energy - balanced arms and wheel movements
        elif energy > 1.0 and brightness > 1.2:
            # Flowing movements
            if duration > 1.0:
                return 'diagonal_drift'  # Base: diagonal mecanum movement + arms
            else:
                return 'flowing_reach'  # Servo: sustained arm movements
                
        elif energy > 1.0 and brightness < 0.8:
            # Deep, rhythmic movements
            if onset > 0.5:
                return 'circular_flow'  # Base: gentle circular movement + arms
            else:
                return 'deep_pulse'  # Servo: rhythmic arm pulses
                
        elif energy > 1.0:
            # Medium energy with mixed movements
            if duration > 0.6:
                return 'sideways_slide'  # Base: lateral sliding + arms
            else:
                return 'dramatic_sweep'  # Servo: dramatic arm sweeps
        
        # Medium energy movements - gentle combinations
        elif brightness > 1.3:
            # Bright, sparkling movements
            if onset > 0.4:
                return 'smooth_glide'  # Base: gentle forward glide + arms
            else:
                return 'bright_sparkle'  # Servo: quick bright arm movements
                
        elif energy > 0.8 and brightness > 0.8:
            # Balanced energy - flowing movements
            return 'flowing_reach'  # Servo: sustained arm movements
        
        # Lower energy movements - gentle arms with subtle wheel movements
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
        """Calculate Mecanum base movement commands - DYNAMIC FAST DANCING"""
        energy = features['energy']
        duration = features['duration']
        brightness = features['brightness']
        onset = features['onset_strength']
        
        # CONSTRAINED SPACE DANCING: Fast but contained movements
        # Reduced speeds for forward/backward to stay in place, keep spins fast
        base_speed = min(1.5, energy * 0.8)  # Max 1.5 m/s - FAST but constrained!
        angular_speed = min(6.0, energy * 3.0)  # Max 6.0 rad/s - FAST SPINS unchanged!
        
        # Add speed boost for high brightness (bright musical passages)
        if brightness > 1.2:
            base_speed *= 1.3  # 30% speed boost for bright sections
            angular_speed *= 1.2  # 20% spin boost
            
        # Add speed boost for strong onsets (musical accents)
        if onset > 0.6:
            base_speed *= 1.2  # 20% speed boost for strong beats
            angular_speed *= 1.4  # 40% spin boost for accents
        
        # Apply TEMPO SCALING for music synchronization
        base_speed *= tempo_scale
        angular_speed *= tempo_scale
        
        # Cap the speeds at maximum after tempo scaling
        base_speed = min(1.5, base_speed)  # Reduced max linear speed for constrained space
        angular_speed = min(6.0, angular_speed)  # Keep fast spins
        
        base_command = {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}
        
        if movement_type == 'sideways_slide':
            # FAST lateral slides - dynamic side steps
            direction = 1 if features['brightness'] > 1.0 else -1
            base_command['linear_y'] = direction * base_speed * 0.9  # Fast lateral movement
            
        elif movement_type == 'diagonal_drift':
            # FAST diagonal movement - signature Mecanum dance move
            forward_speed = base_speed * 0.8  # Fast forward component
            side_speed = base_speed * 0.7  # Fast side component
            side_direction = 1 if features['onset_strength'] > 0.5 else -1
            base_command['linear_x'] = forward_speed
            base_command['linear_y'] = side_direction * side_speed
            
        elif movement_type == 'spin_in_place':
            # COOL FAST SPINS - Dynamic rotations with variations
            spin_direction = 1 if features['brightness'] > features['energy'] else -1
            
            # Dynamic spin variations based on musical features
            if energy > 1.8 and onset > 0.7:
                # MAXIMUM INTENSITY SPIN - Full speed with random direction changes
                if random.random() > 0.7:
                    spin_direction *= -1  # Surprise direction change!
                base_command['angular_z'] = spin_direction * angular_speed  # FULL SPEED!
            elif energy > 1.5:
                # HIGH ENERGY SPIN - Fast with potential direction variation
                if onset > 0.6 and random.random() > 0.8:
                    spin_direction *= -1  # Occasional direction change
                base_command['angular_z'] = spin_direction * angular_speed * 0.9
            elif energy > 1.0:
                # MEDIUM ENERGY SPIN - Good speed
                base_command['angular_z'] = spin_direction * angular_speed * 0.7
            else:
                # CONTROLLED SPIN - Still fast but controlled
                base_command['angular_z'] = spin_direction * angular_speed * 0.5
            
        elif movement_type == 'circular_flow':
            # FAST circular movement - dynamic dance pivot
            base_command['linear_x'] = base_speed * 0.7  # Fast forward
            rotation_direction = 1 if features['onset_strength'] > 0.5 else -1
            base_command['angular_z'] = rotation_direction * angular_speed * 0.6
            
        elif movement_type == 'zigzag_dance':
            # DYNAMIC directional changes - COOL RANDOM dance steps
            direction = random.choice([-1, 1])
            move_choice = random.random()
            
            if onset > 0.7:
                # SUPER DYNAMIC moves for strong beats
                if move_choice > 0.8:
                    # COMBO MOVE: Side + spin
                    base_command['linear_y'] = direction * base_speed * 0.8
                    base_command['angular_z'] = -direction * angular_speed * 0.4
                elif move_choice > 0.6:
                    # COMBO MOVE: Forward + spin
                    base_command['linear_x'] = direction * base_speed * 0.8
                    base_command['angular_z'] = direction * angular_speed * 0.5
                elif move_choice > 0.4:
                    # DIAGONAL DASH
                    base_command['linear_x'] = direction * base_speed * 0.7
                    base_command['linear_y'] = -direction * base_speed * 0.7
                else:
                    # PURE SPIN
                    base_command['angular_z'] = direction * angular_speed * 0.8
            else:
                # STANDARD zigzag moves
                if move_choice > 0.66:
                    # FAST side step
                    base_command['linear_y'] = direction * base_speed * 0.9
                elif move_choice > 0.33:
                    # FAST forward/backward step
                    base_command['linear_x'] = direction * base_speed * 0.9
                else:
                    # FAST spin move
                    base_command['angular_z'] = direction * angular_speed * 0.7
                
        elif movement_type == 'smooth_glide':
            # FAST forward glide - dynamic flow
            base_command['linear_x'] = base_speed * 0.8
            
        elif movement_type == 'rhythmic_steps':
            # FAST rhythmic forward/backward steps - DYNAMIC dance move
            direction = 1 if features['onset_strength'] > 0.6 else -1
            base_command['linear_x'] = direction * base_speed * 0.9
            
        elif movement_type == 'explosive_burst':
            # EXPLOSIVE directional burst - MAXIMUM COOL FACTOR!
            burst_intensity = 1.0
            if onset > 0.8:
                burst_intensity = 1.2  # Even more intense for very strong beats
            
            # Create an extensive list of cool burst moves
            burst_moves = [
                # Basic directional bursts
                {'linear_x': base_speed * burst_intensity, 'linear_y': 0.0},  # FAST forward dash
                {'linear_x': -base_speed * burst_intensity, 'linear_y': 0.0},  # FAST backward dash
                {'linear_x': 0.0, 'linear_y': base_speed * burst_intensity},  # FAST left dash
                {'linear_x': 0.0, 'linear_y': -base_speed * burst_intensity},  # FAST right dash
                
                # Pure spins
                {'angular_z': angular_speed * burst_intensity},  # FAST spin left
                {'angular_z': -angular_speed * burst_intensity},  # FAST spin right
                
                # Diagonal dashes - signature Mecanum moves
                {'linear_x': base_speed * 0.8, 'linear_y': base_speed * 0.8},  # Northeast dash
                {'linear_x': -base_speed * 0.8, 'linear_y': -base_speed * 0.8},  # Southwest dash
                {'linear_x': base_speed * 0.8, 'linear_y': -base_speed * 0.8},  # Southeast dash
                {'linear_x': -base_speed * 0.8, 'linear_y': base_speed * 0.8},  # Northwest dash
                
                # COMBO MOVES - Linear + Angular (MOST COOL!)
                {'linear_x': base_speed * 0.7, 'angular_z': angular_speed * 0.6},  # Forward + spin left
                {'linear_x': base_speed * 0.7, 'angular_z': -angular_speed * 0.6},  # Forward + spin right
                {'linear_x': -base_speed * 0.7, 'angular_z': angular_speed * 0.6},  # Backward + spin left
                {'linear_x': -base_speed * 0.7, 'angular_z': -angular_speed * 0.6},  # Backward + spin right
                {'linear_y': base_speed * 0.7, 'angular_z': angular_speed * 0.6},  # Left + spin
                {'linear_y': -base_speed * 0.7, 'angular_z': -angular_speed * 0.6},  # Right + spin
                
                # COMPLEX COMBOS for very high energy
                {'linear_x': base_speed * 0.6, 'linear_y': base_speed * 0.6, 'angular_z': angular_speed * 0.4},  # Diagonal + spin
                {'linear_x': -base_speed * 0.6, 'linear_y': -base_speed * 0.6, 'angular_z': -angular_speed * 0.4},  # Diagonal back + spin
            ]
            
            # Choose move based on energy level
            if energy > 1.7 and len(burst_moves) > 8:
                # High energy - prefer complex combo moves
                chosen_move = random.choice(burst_moves[8:])  # Complex moves only
            else:
                # Normal energy - any move
                chosen_move = random.choice(burst_moves)
                
            base_command.update(chosen_move)
            
        return base_command

    def calculate_subtle_base_complement(self, movement_type, features, tempo_scale=1.0):
        """Calculate subtle wheel movements to complement servo arm movements"""
        energy = features['energy']
        onset = features['onset_strength']
        
        # Subtle but faster movements that complement arm choreography dynamically
        base_speed = min(0.8, energy * 0.4)  # More dynamic complementary movements
        angular_speed = min(2.0, energy * 1.0)  # Faster complementary rotation
        
        # Apply tempo scaling
        base_speed *= tempo_scale
        angular_speed *= tempo_scale
        
        # Cap the speeds
        base_speed = min(0.8, base_speed)
        angular_speed = min(2.0, angular_speed)
        
        base_command = {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}
        
        # Add subtle wheel movements based on arm movement type
        if movement_type in ['powerful_strike', 'dramatic_sweep']:
            # Slight rotation during dramatic arm movements
            direction = 1 if onset > 0.5 else -1
            base_command['angular_z'] = direction * angular_speed * 0.3
            
        elif movement_type in ['energetic_wave', 'bright_sparkle']:
            # Tiny side shifts during energetic arm movements
            direction = 1 if energy > 1.0 else -1
            base_command['linear_y'] = direction * base_speed * 0.5
            
        elif movement_type in ['flowing_reach', 'subtle_sway']:
            # Very gentle forward/backward movement during flowing arms
            direction = 1 if onset > 0.4 else -1
            base_command['linear_x'] = direction * base_speed * 0.4
            
        elif movement_type == 'deep_pulse':
            # Slight pulsing movement to match arm pulses
            if onset > 0.5:
                base_command['linear_x'] = base_speed * 0.3
            else:
                base_command['linear_x'] = -base_speed * 0.3
                
        elif movement_type == 'gentle_wave':
            # Minimal swaying to complement gentle arm waves
            direction = 1 if features['brightness'] > 1.0 else -1
            base_command['angular_z'] = direction * angular_speed * 0.2
            
        return base_command

    def calculate_complementary_servo_positions(self, movement_type, features, base_amplitude):
        """Calculate subtle servo movements to complement base movements"""
        positions = {}
        
        # Subtle servo movements that complement base motion
        complement_amplitude = base_amplitude * 0.3  # Much gentler than main servo movements
        
        for sid in self.servo_ids:
            if movement_type in ['sideways_slide', 'diagonal_drift']:
                # Gentle wave during lateral movements
                positions[sid] = self.home_positions[sid] + ((-1) ** sid) * complement_amplitude * 0.2
            elif movement_type == 'spin_in_place':
                # Outward reach during spins
                positions[sid] = self.home_positions[sid] + complement_amplitude * 0.4
            elif movement_type in ['circular_flow', 'smooth_glide']:
                # Gentle flowing motion
                positions[sid] = self.home_positions[sid] + random.uniform(-0.2, 0.2) * complement_amplitude
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
        
        self.get_logger().info("Performance started with bulletproof timing system!")

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
            
            # HYBRID DANCING ENABLED - Always send wheel commands for synchronized arms + wheels
            if buffered_movement['base_msg']:
                self.cmd_vel_pub.publish(buffered_movement['base_msg'])
                movement_category = buffered_movement['movement_category']
                if movement_category == 'base':
                    self.get_logger().debug("🕺 Sent primary wheel dance command")
                else:
                    self.get_logger().debug("🤖 Sent complementary wheel movement with arms")
            
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
        
        for _ in range(30):
            self.cmd_vel_pub.publish(stop_twist)
            time.sleep(0.01)
        
        # Wait a moment then return to home
        time.sleep(0.5)
        self.return_to_home()
        
        # Wait to ensure home position is reached
        time.sleep(1.0)
        
        # Send home command again to be absolutely sure
        self.return_to_home()
        
        # FINAL wheel stop command
        for _ in range(10):
            self.cmd_vel_pub.publish(stop_twist)
            time.sleep(0.01)
        
        self.performance_active = False
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
        
        # Create base movement message with QUICK BURST duration constraints
        base_msg = None
        base_command = movement_commands['base_command']
        if any(abs(base_command[key]) > 0.01 for key in ['linear_x', 'linear_y', 'angular_z']):  # Threshold for fast movements
            base_msg = Twist()
            base_msg.linear.x = float(base_command['linear_x'])
            base_msg.linear.y = float(base_command['linear_y'])
            base_msg.angular.z = float(base_command['angular_z'])
            
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
        
        # Send stop commands rapidly and repeatedly
        for _ in range(20):  # More stop commands to ensure wheels stop
            self.cmd_vel_pub.publish(stop_twist)
            time.sleep(0.01)  # Shorter sleep, more frequent
        
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
            
            for _ in range(30):  # Even more stop commands
                self.cmd_vel_pub.publish(stop_twist)
                time.sleep(0.005)  # Very fast publishing
            
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
        
        for _ in range(50):  # MASSIVE number of stop commands
            self.cmd_vel_pub.publish(stop_twist)
            time.sleep(0.002)  # Very rapid fire
        
        # Publish emergency stop signal
        emergency_msg = Bool()
        emergency_msg.data = True
        for _ in range(5):
            self.emergency_stop_pub.publish(emergency_msg)
            time.sleep(0.02)
        
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
        if node:
            node.get_logger().info("Keyboard interrupt - emergency stopping...")
            if hasattr(node, 'emergency_stop'):
                node.emergency_stop()
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
