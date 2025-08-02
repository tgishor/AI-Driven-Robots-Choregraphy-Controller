#!/bin/bash

# Multi-Robot Arm Synchronization Example
# This script demonstrates how to set up synchronized arm control

echo "🦾 Multi-Robot Arm Synchronization Setup"
echo "========================================"

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
echo "🚀 Starting Arm Synchronization..."
echo ""

# Function to run commands in new terminals
run_in_terminal() {
    gnome-terminal -- bash -c "$1; exec bash"
}

echo "1️⃣ Starting Arm Synchronizer (Robot1 → Robot2)"
run_in_terminal "ros2 run jetrover_controller arm_sync --leader robot1 --follower robot2"

sleep 2

echo "2️⃣ Starting Robot1 Arm Controller"
run_in_terminal "ros2 run jetrover_controller multi_arm --robot robot1"

echo ""
echo "✅ Setup Complete!"
echo ""
echo "📋 What's Running:"
echo "   • Arm Synchronizer: robot1 → robot2"
echo "   • Robot1 Arm Controller: Manual control"
echo "   • Robot2: Automatically follows Robot1"
echo ""
echo "🎮 Controls:"
echo "   • Robot1 will run a demo sequence automatically"
echo "   • Robot2 will mirror all movements"
echo "   • Press Ctrl+C in any terminal to stop"
echo ""
echo "🔧 Alternative Commands:"
echo "   # Reverse sync (Robot2 → Robot1):"
echo "   ros2 run jetrover_controller arm_sync --leader robot2 --follower robot1"
echo ""
echo "   # Control individual robots:"
echo "   ros2 run jetrover_controller move_arm --robot robot1"
echo "   ros2 run jetrover_controller move_arm --robot robot2"