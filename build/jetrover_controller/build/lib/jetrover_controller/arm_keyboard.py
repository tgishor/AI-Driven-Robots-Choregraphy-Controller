#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from servo_controller_msgs.msg import ServosPosition, ServoPosition
from servo_controller_msgs.msg import ServoStateList
import sys
import termios
import tty
import select
import time

class ArmKeyboardController(Node):
    def __init__(self, robot_namespace=''):
        # Create unique node name with namespace
        node_name = 'arm_keyboard_controller'
        if robot_namespace:
            node_name = f'arm_keyboard_controller_{robot_namespace}'
            
        super().__init__(node_name)
        
        # Build topic names with namespace
        if robot_namespace:
            servo_controller_topic = f'/{robot_namespace}/servo_controller'
            servo_states_topic = f'/{robot_namespace}/controller_manager/servo_states'
        else:
            servo_controller_topic = 'servo_controller'
            servo_states_topic = '/servo_states'
            
        self.publisher_ = self.create_publisher(ServosPosition, servo_controller_topic, 10)
        self.robot_namespace = robot_namespace
        
        # Current servo positions
        self.current_positions = {
            1: 500,   # joint1
            2: 500,   # joint2  
            3: 500,   # joint3
            4: 500,   # joint4
            5: 500,   # joint5
            10: 500   # gripper
        }
        
        # Movement step size
        self.step_size = 25
        self.duration = 0.2  # Fast response time
        
        # Subscribe to servo states to track current positions
        self.create_subscription(
            ServoStateList,
            servo_states_topic,
            self.servo_state_callback,
            10
        )
        
        # Servo mapping
        self.servo_map = {
            'joint1': 1,
            'joint2': 2,
            'joint3': 3,
            'joint4': 4,
            'joint5': 5,
            'gripper': 10,
        }
        
        # Print control interface
        self.print_interface()
        
    def servo_state_callback(self, msg):
        """Update current servo positions from feedback"""
        for servo in msg.servo_state:
            if servo.id in self.current_positions:
                self.current_positions[servo.id] = servo.position
    
    def print_interface(self):
        """Print the control interface"""
        print("🦾 ARM KEYBOARD CONTROLLER")
        if self.robot_namespace:
            print(f"🏷️  Controlling Robot: {self.robot_namespace}")
        print("="*50)
        print("JOINT CONTROLS:")
        print("  1/! : Joint1 -/+ (Base rotation)")
        print("  2/@ : Joint2 -/+ (Shoulder)")  
        print("  3/# : Joint3 -/+ (Elbow)")
        print("  4/$ : Joint4 -/+ (Wrist1)")
        print("  5/% : Joint5 -/+ (Wrist2)")
        print("  0/) : Gripper -/+ (Open/Close)")
        print("")
        print("PRESETS:")
        print("  h   : Home position (all 500)")
        print("  u   : Reach up")
        print("  f   : Reach forward")
        print("  d   : Reach down")
        print("")
        print("SETTINGS:")
        print("  +/- : Increase/Decrease step size")
        print("  p   : Print current positions")
        print("  r   : Reset to home")
        print("  SPACE: Stop all movement")
        print("  x   : Exit")
        print("="*50)
        print(f"Step size: {self.step_size} pulses")
        print(f"Duration: {self.duration:.1f}s")
        self.print_positions()
        
    def print_positions(self):
        """Print current servo positions"""
        print(f"Current: J1:{self.current_positions[1]:3.0f} J2:{self.current_positions[2]:3.0f} J3:{self.current_positions[3]:3.0f} J4:{self.current_positions[4]:3.0f} J5:{self.current_positions[5]:3.0f} G:{self.current_positions[10]:3.0f}")
        
    def get_key(self):
        """Get single keypress without Enter"""
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)
        return key
        
    def move_servo(self, servo_id, target_position):
        """Move a single servo to target position"""
        # Clamp position to safe range
        target_position = max(50, min(950, target_position))
        
        msg = ServosPosition()
        msg.position_unit = 'pulse'
        msg.duration = self.duration
        
        servo = ServoPosition()
        servo.id = servo_id
        servo.position = float(target_position)
        msg.position.append(servo)
        
        self.publisher_.publish(msg)
        
        # Update our tracking
        self.current_positions[servo_id] = target_position
        
    def move_multiple_servos(self, servo_positions):
        """Move multiple servos at once"""
        msg = ServosPosition()
        msg.position_unit = 'pulse'
        msg.duration = self.duration * 2  # Slower for coordinated moves
        
        for servo_id, position in servo_positions.items():
            # Clamp position to safe range
            position = max(50, min(950, position))
            
            servo = ServoPosition()
            servo.id = servo_id
            servo.position = float(position)
            msg.position.append(servo)
            
            # Update our tracking
            self.current_positions[servo_id] = position
            
        self.publisher_.publish(msg)
        
    def preset_home(self):
        """Move to home position"""
        print("🏠 Moving to HOME position")
        self.move_multiple_servos({
            1: 500,   # joint1
            2: 500,   # joint2
            3: 500,   # joint3
            4: 500,   # joint4
            5: 500,   # joint5
            10: 500   # gripper
        })
        
    def preset_reach_up(self):
        """Move to reach up position"""
        print("⬆️  Moving to REACH UP position")
        self.move_multiple_servos({
            1: 500,   # joint1 - center
            2: 700,   # joint2 - shoulder up
            3: 700,   # joint3 - elbow up
            4: 600,   # joint4 - wrist
            5: 500,   # joint5 - wrist rotate
            10: 400   # gripper - slightly closed
        })
        
    def preset_reach_forward(self):
        """Move to reach forward position"""
        print("➡️  Moving to REACH FORWARD position")
        self.move_multiple_servos({
            1: 500,   # joint1 - center
            2: 300,   # joint2 - shoulder forward
            3: 200,   # joint3 - elbow extended
            4: 200,   # joint4 - wrist forward
            5: 500,   # joint5 - wrist center
            10: 300   # gripper - open
        })
        
    def preset_reach_down(self):
        """Move to reach down position"""
        print("⬇️  Moving to REACH DOWN position")
        self.move_multiple_servos({
            1: 500,   # joint1 - center
            2: 200,   # joint2 - shoulder down
            3: 200,   # joint3 - elbow down
            4: 400,   # joint4 - wrist down
            5: 500,   # joint5 - wrist center
            10: 600   # gripper - closed
        })
        
    def run(self):
        """Main control loop"""
        self.original_settings = termios.tcgetattr(sys.stdin)
        
        try:
            while True:
                key = self.get_key()
                
                # Joint 1 control (Base rotation)
                if key == '1':
                    new_pos = self.current_positions[1] - self.step_size
                    self.move_servo(1, new_pos)
                    print(f"🔧 Joint1: {new_pos:.0f}")
                    
                elif key == '!':
                    new_pos = self.current_positions[1] + self.step_size
                    self.move_servo(1, new_pos)
                    print(f"🔧 Joint1: {new_pos:.0f}")
                    
                # Joint 2 control (Shoulder)
                elif key == '2':
                    new_pos = self.current_positions[2] - self.step_size
                    self.move_servo(2, new_pos)
                    print(f"🔧 Joint2: {new_pos:.0f}")
                    
                elif key == '@':
                    new_pos = self.current_positions[2] + self.step_size
                    self.move_servo(2, new_pos)
                    print(f"🔧 Joint2: {new_pos:.0f}")
                    
                # Joint 3 control (Elbow)
                elif key == '3':
                    new_pos = self.current_positions[3] - self.step_size
                    self.move_servo(3, new_pos)
                    print(f"🔧 Joint3: {new_pos:.0f}")
                    
                elif key == '#':
                    new_pos = self.current_positions[3] + self.step_size
                    self.move_servo(3, new_pos)
                    print(f"🔧 Joint3: {new_pos:.0f}")
                    
                # Joint 4 control (Wrist1)
                elif key == '4':
                    new_pos = self.current_positions[4] - self.step_size
                    self.move_servo(4, new_pos)
                    print(f"🔧 Joint4: {new_pos:.0f}")
                    
                elif key == '$':
                    new_pos = self.current_positions[4] + self.step_size
                    self.move_servo(4, new_pos)
                    print(f"🔧 Joint4: {new_pos:.0f}")
                    
                # Joint 5 control (Wrist2)
                elif key == '5':
                    new_pos = self.current_positions[5] - self.step_size
                    self.move_servo(5, new_pos)
                    print(f"🔧 Joint5: {new_pos:.0f}")
                    
                elif key == '%':
                    new_pos = self.current_positions[5] + self.step_size
                    self.move_servo(5, new_pos)
                    print(f"🔧 Joint5: {new_pos:.0f}")
                    
                # Gripper control
                elif key == '0':
                    new_pos = self.current_positions[10] - self.step_size
                    self.move_servo(10, new_pos)
                    print(f"🤏 Gripper: {new_pos:.0f} (Open)")
                    
                elif key == ')':
                    new_pos = self.current_positions[10] + self.step_size
                    self.move_servo(10, new_pos)
                    print(f"🤏 Gripper: {new_pos:.0f} (Close)")
                    
                # Presets
                elif key == 'h':
                    self.preset_home()
                    
                elif key == 'u':
                    self.preset_reach_up()
                    
                elif key == 'f':
                    self.preset_reach_forward()
                    
                elif key == 'd':
                    self.preset_reach_down()
                    
                elif key == 'r':
                    self.preset_home()
                    
                # Settings
                elif key == '+' or key == '=':
                    self.step_size = min(100, self.step_size + 5)
                    print(f"⚡ Step size: {self.step_size}")
                    
                elif key == '-':
                    self.step_size = max(5, self.step_size - 5)
                    print(f"🐌 Step size: {self.step_size}")
                    
                elif key == 'p':
                    self.print_positions()
                    
                elif key == ' ':
                    print("🛑 STOP - Current positions maintained")
                    # Just print current positions, no movement
                    self.print_positions()
                    
                elif key == 'x' or key == '\x03':  # x or Ctrl+C
                    print("👋 Exiting...")
                    break
                    
                else:
                    print("❓ Unknown key. Press 'x' to exit, 'p' for positions.")
                    
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)

def main(args=None):
    rclpy.init(args=args)
    
    # Parse remaining arguments for robot namespace
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
        controller = ArmKeyboardController(robot_namespace)
        controller.run()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()