# Multi-Robot Arm Synchronization

This implementation provides synchronized arm control for multiple robots in the JetRover system.

## Components

### 1. ArmKeyboardController (`arm_keyboard.py`) 🆕
- **Real-time keyboard control for individual servos/joints**
- Similar to the robot movement keyboard teleop
- Control each joint individually with dedicated keys
- Includes preset positions and step-size adjustment

### 2. MultiRobotArmController (`multi_robot_arm_controller.py`)
- Controls a specific robot's arm using namespace
- Similar to the keyboard teleop pattern
- Runs demo sequences and provides arm control interface

### 3. ArmSynchronizer (`arm_synchronizer.py`)
- Synchronizes arm movements between robots
- Listens to one robot's arm commands and forwards them to another
- Similar to the robot movement synchronizer

### 4. Updated ArmController (`move_arm.py`)
- Now supports multi-robot namespaces
- Compatible with the existing single-robot setup

## Usage Examples

### 🎮 Real-time Keyboard Control (Individual Joints)
```bash
# Control Robot1's arm with keyboard (like keyboard teleop for movement)
ros2 run jetrover_controller arm_keyboard --robot robot1

# Control Robot2's arm with keyboard  
ros2 run jetrover_controller arm_keyboard --robot robot2

# Control without namespace (single robot setup)
ros2 run jetrover_controller arm_keyboard
```

### 🎬 Demo Sequences
```bash
# Run the multi-robot arm controller for robot1
ros2 run jetrover_controller multi_arm --robot robot1

# Or use the updated move_arm with namespace
ros2 run jetrover_controller move_arm --robot robot1
```

### 🤖 Multi-Robot Control
```bash
ros2 run jetrover_controller multi_arm --robot robot2
ros2 run jetrover_controller move_arm --robot robot2
```

### 🔄 Synchronized Control with Keyboard
```bash
# Start the arm synchronizer (Robot1 leads, Robot2 follows)
ros2 run jetrover_controller arm_sync --leader robot1 --follower robot2

# Then control robot1's arm with keyboard - robot2 will mirror in real-time!
ros2 run jetrover_controller arm_keyboard --robot robot1
```

### 🔄 Demo Synchronization 
```bash
# Start the arm synchronizer
ros2 run jetrover_controller arm_sync --leader robot1 --follower robot2

# Then run demo sequences on robot1, robot2 will follow automatically
ros2 run jetrover_controller multi_arm --robot robot1
```

### 🔄 Alternative Synchronization (Robot2 leads, Robot1 follows)
```bash
ros2 run jetrover_controller arm_sync --leader robot2 --follower robot1
ros2 run jetrover_controller arm_keyboard --robot robot2
```

## Topic Structure

Each robot publishes/subscribes to namespaced topics:

### Robot1 Topics:
- **Servo Commands**: `/robot1/servo_controller`
- **Servo States**: `/robot1/controller_manager/servo_states`

### Robot2 Topics:
- **Servo Commands**: `/robot2/servo_controller`
- **Servo States**: `/robot2/controller_manager/servo_states`

## Synchronization Flow

1. **Leader Robot**: Controlled manually via `multi_arm` command
2. **ArmSynchronizer**: Listens to leader's servo commands
3. **Follower Robot**: Receives forwarded commands automatically

```
Manual Control → Robot1/servo_controller → ArmSynchronizer → Robot2/servo_controller
```

## 🎮 Keyboard Controls (arm_keyboard)

When using `ros2 run jetrover_controller arm_keyboard`, you get real-time control:

### Joint Controls:
- **1/!** : Joint1 -/+ (Base rotation)
- **2/@** : Joint2 -/+ (Shoulder)  
- **3/#** : Joint3 -/+ (Elbow)
- **4/$** : Joint4 -/+ (Wrist1)
- **5/%** : Joint5 -/+ (Wrist2)
- **0/)** : Gripper -/+ (Open/Close)

### Presets:
- **h** : Home position (all servos to 500)
- **u** : Reach up
- **f** : Reach forward  
- **d** : Reach down

### Settings:
- **+/-** : Increase/Decrease step size (5-100 pulses)
- **p** : Print current positions
- **r** : Reset to home
- **SPACE** : Stop/Show current positions
- **x** : Exit

## Integration with Existing System

This follows the same namespace pattern as:
- `keyboard.py` (multi-robot teleop)
- `robot_synchronizer.py` (movement sync)

The arm synchronization can be used alongside movement synchronization for full robot coordination.

## Demo Sequence

The `MultiRobotArmController` includes a demo sequence that moves through:
1. Home position (all servos at 500 pulses)
2. Extended position (reaching forward)
3. Reach up position (arm raised)
4. Return to home

This sequence runs automatically when starting the controller.