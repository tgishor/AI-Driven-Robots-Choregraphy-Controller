#!/usr/bin/env python3
import os
import time
import threading
import subprocess
import argparse
import random
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from servo_controller_msgs.msg import ServosPosition, ServoPosition, ServoStateList

import librosa
import numpy as np


class MecanumDanceConductor(Node):
    def __init__(
        self,
        audio_path: str,
        robot_namespaces: List[str],
        audio_player: str = 'mpg123',
        duo_mode: str = 'mirror',  # mirror | canon | complementary
        seed: int = 7,
    ) -> None:
        super().__init__('mecanum_dance_conductor')

        random.seed(seed)

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.audio_path = audio_path
        self.audio_player = audio_player
        self.robot_namespaces = robot_namespaces
        self.duo_mode = duo_mode

        # QoS with low latency and reliable delivery (consistent with existing nodes)
        self.realtime_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        # Publishers per robot
        self.cmd_vel_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        self.servo_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        self.current_servo_positions: Dict[str, Dict[int, float]] = {}
        self.pose_received: Dict[str, bool] = {}

        for ns in self.robot_namespaces:
            cmd_topic = f'/{ns}/controller/cmd_vel'
            servo_topic = f'/{ns}/servo_controller'
            servo_states_topic = f'/{ns}/controller_manager/servo_states'

            self.cmd_vel_publishers[ns] = self.create_publisher(Twist, cmd_topic, self.realtime_qos)
            self.servo_publishers[ns] = self.create_publisher(ServosPosition, servo_topic, self.realtime_qos)
            self.current_servo_positions[ns] = {1: 500, 2: 500, 3: 500, 4: 500, 5: 500, 10: 500}
            self.pose_received[ns] = False

            # Subscribe to servo states for each robot
            self.create_subscription(
                ServoStateList,
                servo_states_topic,
                lambda msg, ns=ns: self._servo_state_callback(ns, msg),
                self.realtime_qos,
            )

        # Movement and gesture configuration
        self.max_linear_speed_mps = 0.22  # conservative for indoor choreo
        self.max_lateral_speed_mps = 0.22
        self.max_angular_speed_rps = 1.2
        self.max_servo_delta = 180.0  # pulses around home per beat for expressive gestures
        self.servo_ids = [1, 2, 3, 4, 5, 10]

        # Analysis outputs
        self.sections: List[Tuple[float, float, float]] = []  # (start, end, loud_ratio)
        self.beat_duration_s: float = 0.5

        # Analyze audio and prepare schedule
        self._analyze_audio()

        # Start execution thread
        self.execution_thread = threading.Thread(target=self._conduct, daemon=True)
        self.execution_thread.start()

        robots_str = ', '.join(self.robot_namespaces)
        self.get_logger().info(f"🎵 Mecanum Dance Conductor ready for: {robots_str}")
        self.get_logger().info(f"    Duo mode: {self.duo_mode}")

    def _servo_state_callback(self, robot_ns: str, msg: ServoStateList) -> None:
        for s in msg.servo_state:
            if s.id in self.current_servo_positions[robot_ns]:
                self.current_servo_positions[robot_ns][s.id] = float(s.position)
        self.pose_received[robot_ns] = True

    def _analyze_audio(self) -> None:
        self.get_logger().info("Analyzing audio beats and dynamics…")
        y, sr = librosa.load(self.audio_path, sr=None)
        tempo_bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        median_onset = float(np.median(onset_env)) or 1.0

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if len(beat_times) < 2:
            # Fallback to 120 BPM grid
            self.beat_duration_s = 0.5
            total_duration_s = librosa.get_duration(y=y, sr=sr)
            num_beats = int(total_duration_s / self.beat_duration_s)
            for i in range(num_beats - 1):
                s = i * self.beat_duration_s
                e = (i + 1) * self.beat_duration_s
                self.sections.append((s, e, 1.0))
            return

        # Beat duration from median
        beat_intervals = np.diff(beat_times)
        self.beat_duration_s = float(np.median(beat_intervals))

        for i in range(len(beat_times) - 1):
            start = float(beat_times[i])
            end = float(beat_times[i + 1])
            s_frame = librosa.time_to_frames(start, sr=sr)
            e_frame = librosa.time_to_frames(end, sr=sr)
            loud = float(np.mean(onset_env[s_frame:e_frame])) if e_frame > s_frame else median_onset
            loud_ratio = max(0.2, min(2.0, loud / median_onset))  # 0.2..2.0
            self.sections.append((start, end, loud_ratio))

    # ------------------------- Conducting -------------------------
    def _conduct(self) -> None:
        # Wait briefly for servo states
        t0 = time.time()
        while time.time() - t0 < 5.0 and not all(self.pose_received.values()):
            time.sleep(0.05)

        # Start audio playback
        audio_thread = threading.Thread(target=self._play_audio, daemon=True)
        audio_thread.start()
        start_time = time.time() + 0.6  # small lead to align first command

        # Phrase planning: group beats into phrases with a single choreo motif
        phrase_len_beats = 4
        patterns = [
            'sway',
            'strafe_mirror',
            'diagonal_box',
            'spin_and_wave',
            'circle_pair',
            'criss_cross',
            'pose_hold',
        ]

        for phrase_start in range(0, len(self.sections), phrase_len_beats):
            phrase_sections = self.sections[phrase_start:phrase_start + phrase_len_beats]
            if not phrase_sections:
                break

            pattern = random.choice(patterns)

            # Execute the phrase beat by beat
            for local_idx, (s, e, loud_ratio) in enumerate(phrase_sections):
                scheduled_at = start_time + s
                wait_s = scheduled_at - time.time()
                if wait_s > 0:
                    time.sleep(wait_s)

                duration = max(0.15, e - s)
                energy = min(1.0, (loud_ratio - 0.2) / 1.8)  # 0..1

                # Publish for each robot with per-robot role/variation
                for idx, ns in enumerate(self.robot_namespaces):
                    role = self._role_for_robot(idx, local_idx)
                    self._execute_pattern_for_robot(ns, pattern, role, energy, duration)

        # End: stop motions and return arms to home
        self._all_stop()
        self._all_return_home(2.0)

    def _role_for_robot(self, robot_index: int, beat_index_in_phrase: int) -> str:
        if self.duo_mode == 'mirror':
            return 'A' if robot_index == 0 else 'B'
        if self.duo_mode == 'canon':
            # Off-by-one-beat echo for robot 2
            return 'A' if (robot_index == 0 or beat_index_in_phrase % 2 == 0) else 'B'
        # complementary
        return 'A' if robot_index % 2 == 0 else 'B'

    # ------------------------- Pattern Execution -------------------------
    def _execute_pattern_for_robot(self, ns: str, pattern: str, role: str, energy: float, duration: float) -> None:
        # Body movement
        twist = self._twist_for_pattern(pattern, role, energy)
        self.cmd_vel_publishers[ns].publish(twist)

        # Arm gesture coordinated with body
        targets = self._arm_targets_for_pattern(ns, pattern, role, energy)
        self._publish_servos(ns, duration * 0.9, targets)

        # If canon mode and role is B, optionally delay a hair (already staggered at role selection)

    def _twist_for_pattern(self, pattern: str, role: str, energy: float) -> Twist:
        lin_scale = self.max_linear_speed_mps * energy
        lat_scale = self.max_lateral_speed_mps * energy
        ang_scale = self.max_angular_speed_rps * energy

        t = Twist()

        if pattern == 'sway':
            # Sideways sway on beat
            direction = 1.0 if role == 'A' else -1.0
            t.linear.y = direction * lat_scale
            t.angular.z = 0.15 * direction * ang_scale
            return t

        if pattern == 'strafe_mirror':
            # Mirror strafe left/right
            direction = 1.0 if role == 'A' else -1.0
            t.linear.y = direction * lat_scale
            return t

        if pattern == 'diagonal_box':
            # Box step across 4 beats, mapped by role to diagonals
            # Role determines primary axis; alternate slight rotation
            t.linear.x = lin_scale * (1.0 if role == 'A' else -1.0)
            t.linear.y = lat_scale * (1.0 if role == 'A' else 1.0)
            t.angular.z = 0.2 * (-ang_scale if role == 'A' else ang_scale)
            return t

        if pattern == 'spin_and_wave':
            t.angular.z = (-ang_scale if role == 'A' else ang_scale)
            t.linear.x = 0.05 * lin_scale
            return t

        if pattern == 'circle_pair':
            # Small circles in opposite directions
            t.linear.x = 0.6 * lin_scale
            t.linear.y = (lat_scale if role == 'A' else -lat_scale)
            t.angular.z = (0.3 * ang_scale if role == 'A' else -0.3 * ang_scale)
            return t

        if pattern == 'criss_cross':
            # Cross paths: one forward-left, other forward-right
            t.linear.x = 0.7 * lin_scale
            t.linear.y = (lat_scale if role == 'A' else -lat_scale)
            return t

        if pattern == 'pose_hold':
            # Minimal movement
            t.linear.x = 0.0
            t.linear.y = 0.0
            t.angular.z = 0.0
            return t

        return t

    def _arm_targets_for_pattern(self, ns: str, pattern: str, role: str, energy: float) -> Dict[int, float]:
        # Use current position as base; fall back to home 500
        home = self.current_servo_positions.get(ns, {i: 500.0 for i in self.servo_ids})
        amp = self.max_servo_delta * (0.4 + 0.6 * energy)

        def clamp(x: float) -> float:
            return float(max(50.0, min(950.0, x)))

        targets: Dict[int, float] = {}

        if pattern == 'sway':
            # Shoulder-elbow sway with subtle wrist twist
            delta = amp * (1.0 if role == 'A' else -1.0)
            targets[2] = clamp(home[2] + 0.25 * delta)
            targets[3] = clamp(home[3] - 0.25 * delta)
            targets[4] = clamp(home[4] + 0.15 * delta)
            targets[5] = clamp(home[5] - 0.10 * delta)
            targets[10] = clamp(home[10] + 0.05 * delta)
            return targets

        if pattern == 'strafe_mirror':
            # Base rotation small, wrist counter
            delta = amp * (1.0 if role == 'A' else -1.0)
            targets[1] = clamp(home[1] + 0.2 * delta)
            targets[4] = clamp(home[4] - 0.15 * delta)
            targets[5] = clamp(home[5] + 0.15 * delta)
            return targets

        if pattern == 'diagonal_box':
            # Reach-forward vs reach-up alternating by role
            if role == 'A':
                targets[2] = clamp(home[2] - 0.35 * amp)
                targets[3] = clamp(home[3] - 0.30 * amp)
                targets[4] = clamp(home[4] - 0.25 * amp)
                targets[10] = clamp(home[10] - 0.10 * amp)
            else:
                targets[2] = clamp(home[2] + 0.35 * amp)
                targets[3] = clamp(home[3] + 0.30 * amp)
                targets[4] = clamp(home[4] + 0.15 * amp)
                targets[5] = clamp(home[5] - 0.10 * amp)
            return targets

        if pattern == 'spin_and_wave':
            # Role A spins, Role B waves big
            if role == 'B':
                # Wave: oscillate wrist and elbow; here just target a high pose
                targets[2] = clamp(home[2] + 0.40 * amp)
                targets[3] = clamp(home[3] - 0.40 * amp)
                targets[4] = clamp(home[4] + 0.50 * amp)
                targets[5] = clamp(home[5] + 0.30 * amp)
            else:
                # Spin pose: compact arms
                targets[2] = clamp(home[2] + 0.10 * amp)
                targets[3] = clamp(home[3] + 0.10 * amp)
                targets[4] = clamp(home[4] - 0.15 * amp)
                targets[10] = clamp(home[10] + 0.10 * amp)
            return targets

        if pattern == 'circle_pair':
            # Opposing tilt
            delta = amp * (1.0 if role == 'A' else -1.0)
            targets[1] = clamp(home[1] + 0.25 * delta)
            targets[2] = clamp(home[2] - 0.20 * delta)
            targets[3] = clamp(home[3] + 0.20 * delta)
            targets[4] = clamp(home[4] - 0.10 * delta)
            return targets

        if pattern == 'criss_cross':
            # One reaches left, other right
            sign = 1.0 if role == 'A' else -1.0
            targets[1] = clamp(home[1] + 0.30 * sign * amp)
            targets[2] = clamp(home[2] - 0.25 * amp)
            targets[3] = clamp(home[3] + 0.20 * amp)
            targets[4] = clamp(home[4] - 0.10 * sign * amp)
            return targets

        if pattern == 'pose_hold':
            # Strong static pose; open or close gripper subtly per role
            sign = 1.0 if role == 'A' else -1.0
            targets[2] = clamp(home[2] + 0.20 * sign * amp)
            targets[3] = clamp(home[3] - 0.20 * sign * amp)
            targets[5] = clamp(home[5] + 0.10 * sign * amp)
            targets[10] = clamp(home[10] - 0.10 * sign * amp)
            return targets

        return {}

    # ------------------------- Helpers -------------------------
    def _publish_servos(self, ns: str, duration_s: float, targets: Dict[int, float]) -> None:
        if not targets:
            return
        msg = ServosPosition()
        msg.position_unit = 'pulse'
        msg.duration = float(max(0.05, duration_s))
        for sid, pos in targets.items():
            msg.position.append(ServoPosition(id=int(sid), position=float(pos)))
            self.current_servo_positions[ns][sid] = float(pos)
        self.servo_publishers[ns].publish(msg)

    def _play_audio(self) -> None:
        cmd = [self.audio_player, '-q', self.audio_path] if self.audio_player in ('mpg123', 'mpg321') else [self.audio_player, self.audio_path]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            p.wait()
        except Exception as exc:
            self.get_logger().error(f"Audio playback error: {exc}")

    def _all_stop(self) -> None:
        stop = Twist()
        for pub in self.cmd_vel_publishers.values():
            pub.publish(stop)

    def _all_return_home(self, duration_s: float) -> None:
        for ns in self.robot_namespaces:
            home_targets = {sid: 500.0 for sid in self.servo_ids}
            self._publish_servos(ns, duration_s, home_targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Mecanum duo dance conductor (multi-robot, body+arms)')
    parser.add_argument('--audio', required=True, help='Path to the audio file')
    parser.add_argument('--robots', nargs='+', required=True, help='Robot namespaces, e.g. robot1 robot2')
    parser.add_argument('--player', default='mpg123', choices=['mpg123', 'mpg321', 'mplayer', 'aplay'], help='Audio player')
    parser.add_argument('--duo-mode', default='mirror', choices=['mirror', 'canon', 'complementary'], help='Relationship between robots')
    parser.add_argument('--seed', type=int, default=7, help='Random seed for repeatability')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    try:
        node = MecanumDanceConductor(
            audio_path=args.audio,
            robot_namespaces=args.robots,
            audio_player=args.player,
            duo_mode=args.duo_mode,
            seed=args.seed,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()


