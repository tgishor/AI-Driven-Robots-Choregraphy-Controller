#!/bin/bash

# Simple test script for arm keyboard controller

echo "🦾 Testing Arm Keyboard Controller"
echo "=================================="

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
echo "🚀 Starting Arm Keyboard Controller..."
echo ""

# Prompt user for robot selection
echo "Which robot do you want to control?"
echo "1) robot1"
echo "2) robot2" 
echo "3) no namespace (single robot)"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "🤖 Starting arm keyboard control for robot1..."
        ros2 run jetrover_controller arm_keyboard --robot robot1
        ;;
    2)
        echo "🤖 Starting arm keyboard control for robot2..."
        ros2 run jetrover_controller arm_keyboard --robot robot2
        ;;
    3)
        echo "🤖 Starting arm keyboard control (no namespace)..."
        ros2 run jetrover_controller arm_keyboard
        ;;
    *)
        echo "❌ Invalid choice. Exiting..."
        exit 1
        ;;
esac