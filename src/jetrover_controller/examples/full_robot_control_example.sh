#!/bin/bash

# Complete Multi-Robot Control Example
# Demonstrates synchronized movement AND arm control

echo "🚀 Complete Multi-Robot Control Setup"
echo "====================================="

# Check if we're in the correct directory
if [ ! -f "setup.py" ]; then
    echo "❌ Please run this script from the jetrover_controller package directory"
    exit 1
fi

echo "📦 Building the package..."
cd ../../.. # Go to workspace root
colcon build --packages-select jetrover_controller
source install/setup.bash

echo ""
echo "🚀 Starting Complete Robot Synchronization..."
echo ""

# Function to run commands in new terminals
run_in_terminal() {
    gnome-terminal --tab --title="$2" -- bash -c "$1; exec bash"
}

echo "1️⃣ Starting Movement Synchronizer (Robot1 → Robot2)"
run_in_terminal "ros2 run jetrover_controller robot_sync" "Movement Sync"

sleep 1

echo "2️⃣ Starting Arm Synchronizer (Robot1 → Robot2)"
run_in_terminal "ros2 run jetrover_controller arm_sync --leader robot1 --follower robot2" "Arm Sync"

sleep 1

echo "3️⃣ Starting Robot1 Movement Control (Keyboard)"
run_in_terminal "ros2 run jetrover_controller keyboard --robot robot1" "Robot1 Movement"

sleep 1

echo "4️⃣ Starting Robot1 Arm Control (Keyboard)"
run_in_terminal "ros2 run jetrover_controller arm_keyboard --robot robot1" "Robot1 Arm"

echo ""
echo "✅ Complete Setup Ready!"
echo ""
echo "📋 What's Running:"
echo "   • Movement Sync: robot1 → robot2"
echo "   • Arm Sync: robot1 → robot2"
echo "   • Robot1 Movement: Keyboard control (w/a/s/d/q/e/etc.)"
echo "   • Robot1 Arm: Keyboard control (1-5 for joints, 0 for gripper)"
echo "   • Robot2: Automatically follows ALL Robot1 actions"
echo ""
echo "🎮 You now have FULL control of Robot1:"
echo "   📍 Movement Control Tab:"
echo "     w/s: Forward/Backward    q/e: Rotate L/R"
echo "     a/d: Strafe Left/Right   +/-: Speed control"
echo "     SPACE: Stop              x: Exit"
echo ""
echo "   🦾 Arm Control Tab:"
echo "     1/!: Joint1 (Base)       4/$: Joint4 (Wrist1)"
echo "     2/@: Joint2 (Shoulder)   5/%: Joint5 (Wrist2)"
echo "     3/#: Joint3 (Elbow)      0/): Gripper"
echo "     h: Home  u: Up  f: Forward  d: Down"
echo "     +/-: Step size           x: Exit"
echo ""
echo "🤖 Robot2 will mirror EVERYTHING Robot1 does!"
echo ""
echo "🛑 To stop everything: Press Ctrl+C in each terminal tab"
echo ""
echo "🔧 Individual Control Commands:"
echo "   # Control only movement:"
echo "   ros2 run jetrover_controller keyboard --robot robot1"
echo "   ros2 run jetrover_controller keyboard --robot robot2"
echo ""
echo "   # Control only arms:"
echo "   ros2 run jetrover_controller arm_keyboard --robot robot1"
echo "   ros2 run jetrover_controller arm_keyboard --robot robot2"