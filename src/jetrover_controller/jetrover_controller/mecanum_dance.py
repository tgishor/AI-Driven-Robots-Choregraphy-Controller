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
        compact: bool = True,
        arena_width_m: float = 2.0,
        arena_depth_m: float = 1.5,
        max_disp_per_beat_m: float = 0.12,
        max_yaw_per_beat_rad: float = 0.35,
        beat_pulse_fraction: float = 0.33,
    ) -> None:
        super().__init__('mecanum_dance_conductor')

        random.seed(seed)

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.audio_path = audio_path
        self.audio_player = audio_player
        self.robot_namespaces = robot_namespaces
        self.duo_mode = duo_mode
        self.compact = compact
        self.arena_width_m = max(0.5, float(arena_width_m))
        self.arena_depth_m = max(0.5, float(arena_depth_m))
        self.max_disp_per_beat_m = max(0.01, float(max_disp_per_beat_m))
        self.max_yaw_per_beat_rad = max(0.05, float(max_yaw_per_beat_rad))
        self.beat_pulse_fraction = float(min(0.8, max(0.15, beat_pulse_fraction)))

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
        self.sign_state: Dict[str, int] = {}
        self.virtual_pose: Dict[str, Dict[str, float]] = {}

        for ns in self.robot_namespaces:
            cmd_topic = f'/{ns}/controller/cmd_vel'
            servo_topic = f'/{ns}/servo_controller'
            servo_states_topic = f'/{ns}/controller_manager/servo_states'

            self.cmd_vel_publishers[ns] = self.create_publisher(Twist, cmd_topic, self.realtime_qos)
            self.servo_publishers[ns] = self.create_publisher(ServosPosition, servo_topic, self.realtime_qos)
            self.current_servo_positions[ns] = {1: 500, 2: 500, 3: 500, 4: 500, 5: 500, 10: 500}
            self.pose_received[ns] = False
            self.sign_state[ns] = 1
            self.virtual_pose[ns] = {"x": 0.0, "y": 0.0, "theta": 0.0}

            # Subscribe to servo states for each robot
            self.create_subscription(
                ServoStateList,
                servo_states_topic,
                lambda msg, ns=ns: self._servo_state_callback(ns, msg),
                self.realtime_qos,
            )

        # Movement and gesture configuration
        # Base speed caps (further constrained per-beat by max_disp_per_beat)
        self.max_linear_speed_mps = 0.16 if compact else 0.22
        self.max_lateral_speed_mps = 0.16 if compact else 0.22
        self.max_angular_speed_rps = 0.9 if compact else 1.2
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
        # Small lead to align first command, then use perf_counter for precision
        start_time = time.perf_counter() + 0.35

        # Phrase planning: group beats into phrases with a single choreo motif
        phrase_len_beats = 4

        for phrase_start in range(0, len(self.sections), phrase_len_beats):
            phrase_sections = self.sections[phrase_start:phrase_start + phrase_len_beats]
            if not phrase_sections:
                break

            # Adaptive pattern per beat based on loudness and phrase context
            pattern = self._select_pattern(phrase_sections, local_idx)

            # Execute the phrase beat by beat
            for local_idx, (s, e, loud_ratio) in enumerate(phrase_sections):
                scheduled_at = start_time + s
                now = time.perf_counter()
                wait_s = scheduled_at - now
                if wait_s > 0:
                    time.sleep(wait_s)

                duration = max(0.15, e - s)
                energy = min(1.0, (loud_ratio - 0.2) / 1.8)  # 0..1

                # Compute pulse duration and publish body+arms per robot
                pulse_dur = self.beat_pulse_fraction * duration

                twists: Dict[str, Twist] = {}
                targets_per_robot: Dict[str, Dict[int, float]] = {}

                for idx, ns in enumerate(self.robot_namespaces):
                    role = self._role_for_robot(idx, local_idx)
                    base_twist = self._twist_for_pattern(pattern, role, energy)
                    twist = self._apply_constraints(ns, base_twist, pulse_dur)
                    twists[ns] = twist
                    targets_per_robot[ns] = self._arm_targets_for_pattern(ns, pattern, role, energy)

                # Publish body moves
                for ns, twist in twists.items():
                    self.cmd_vel_publishers[ns].publish(twist)

                # Publish arm gestures
                for ns, targets in targets_per_robot.items():
                    self._publish_servos(ns, duration * 0.85, targets)

                # Hold pulse then stop body
                time.sleep(max(0.05, pulse_dur))
                self._all_stop()

                # Update virtual pose with executed pulse
                for ns, twist in twists.items():
                    self._integrate_virtual_pose(ns, twist, pulse_dur)

        # End: stop motions and return arms to home
        self._all_stop()
        self._all_return_home(2.0)

    def _select_pattern(self, phrase_sections: List[Tuple[float, float, float]], local_idx: int) -> str:
        # Loudness of current beat
        _, _, loud_ratio = phrase_sections[local_idx]
        if loud_ratio > 1.3:
            return 'spin_and_wave' if local_idx % 2 == 0 else 'sway'
        if loud_ratio < 0.7:
            return 'pose_hold' if local_idx % 2 == 0 else 'sway'
        return 'strafe_mirror' if local_idx % 2 == 0 else 'sway'

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

        # Remove wide-ranging diagonal_box/circle/criss patterns for constrained spaces

        if pattern == 'spin_and_wave':
            t.angular.z = (-ang_scale if role == 'A' else ang_scale)
            t.linear.x = 0.05 * lin_scale
            return t

        # Fallback minimal forward pulse on unknown

        if pattern == 'pose_hold':
            # Minimal movement
            t.linear.x = 0.0
            t.linear.y = 0.0
            t.angular.z = 0.0
            return t

        return t

    def _apply_constraints(self, ns: str, twist: Twist, pulse_dur: float) -> Twist:
        # Alternate direction each beat to keep net-zero trend
        sgn = self.sign_state[ns]
        self.sign_state[ns] *= -1

        twist.linear.x *= sgn
        twist.linear.y *= sgn
        twist.angular.z *= sgn

        # Scale to respect per-beat displacement and yaw caps
        abs_lin = abs(twist.linear.x)
        abs_lat = abs(twist.linear.y)
        abs_ang = abs(twist.angular.z)

        def scale_component(value: float, max_per_beat: float) -> float:
            if value == 0.0:
                return 0.0
            max_speed = max_per_beat / max(0.05, pulse_dur)
            scale = min(1.0, max_speed / abs(value))
            return value * scale

        twist.linear.x = scale_component(twist.linear.x, self.max_disp_per_beat_m)
        twist.linear.y = scale_component(twist.linear.y, self.max_disp_per_beat_m)
        twist.angular.z = scale_component(twist.angular.z, self.max_yaw_per_beat_rad)

        # Arena boundary bias: if near edge, bias inward by reducing outward component
        pose = self.virtual_pose[ns]
        half_w = self.arena_width_m / 2.0
        half_d = self.arena_depth_m / 2.0

        # Predict next position
        next_x = pose["x"] + twist.linear.x * pulse_dur
        next_y = pose["y"] + twist.linear.y * pulse_dur

        if next_x > 0.8 * half_d:
            twist.linear.x = min(0.0, twist.linear.x)
        if next_x < -0.8 * half_d:
            twist.linear.x = max(0.0, twist.linear.x)
        if next_y > 0.8 * half_w:
            twist.linear.y = min(0.0, twist.linear.y)
        if next_y < -0.8 * half_w:
            twist.linear.y = max(0.0, twist.linear.y)

        return twist

    def _integrate_virtual_pose(self, ns: str, twist: Twist, dt: float) -> None:
        pose = self.virtual_pose[ns]
        pose["x"] += float(twist.linear.x) * dt
        pose["y"] += float(twist.linear.y) * dt
        pose["theta"] += float(twist.angular.z) * dt

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
    parser.add_argument('--compact', action='store_true', default=True, help='Constrain moves to compact micro-pulses (default on)')
    parser.add_argument('--arena-width', type=float, default=2.0, help='Arena width (Y), meters')
    parser.add_argument('--arena-depth', type=float, default=1.5, help='Arena depth (X), meters')
    parser.add_argument('--max-disp-per-beat', type=float, default=0.12, help='Max linear displacement per beat (m)')
    parser.add_argument('--max-yaw-per-beat', type=float, default=0.35, help='Max yaw per beat (rad)')
    parser.add_argument('--beat-pulse', type=float, default=0.33, help='Fraction of beat used to move, rest to stop')
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
            compact=args.compact,
            arena_width_m=args.arena_width,
            arena_depth_m=args.arena_depth,
            max_disp_per_beat_m=args.max_disp_per_beat,
            max_yaw_per_beat_rad=args.max_yaw_per_beat,
            beat_pulse_fraction=args.beat_pulse,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()


