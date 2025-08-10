#!/usr/bin/env python3
"""
Test script for Advanced Multi-Phase Choreography

This script demonstrates how to use the advanced choreography system
with position tracking and alignment calculations.

Usage:
1. Run this script to see the choreography creation process
2. The script will create a sample choreography file
3. You can then use the keyboard controller to play it

Example choreography:
- Primary time: 3 seconds
- Secondary time: 3 seconds  
- Speed: 0.2 m/s
- Robot 1: Right→Left
- Robot 2: Left→Right
- Automatic alignment at the end
"""

import json
import os
from datetime import datetime

def create_sample_choreography():
    """Create a sample advanced multi-phase choreography"""
    
    # Sample choreography data
    choreography_data = {
        "sample_advanced_move": {
            "type": "advanced_multi_phase",
            "primary_time": 3.0,
            "secondary_time": 3.0,
            "speed": 0.2,
            "timestamp": datetime.now().isoformat(),
            "robot_namespace": "robot1"
        }
    }
    
    # Save to file
    filename = "choreography_robot1.json"
    with open(filename, 'w') as f:
        json.dump(choreography_data, f, indent=2)
    
    print(f"✅ Created sample choreography: {filename}")
    print("📋 Choreography Details:")
    print(f"   Name: sample_advanced_move")
    print(f"   Type: advanced_multi_phase")
    print(f"   Primary time: 3.0 seconds")
    print(f"   Secondary time: 3.0 seconds")
    print(f"   Speed: 0.2 m/s")
    print(f"   Robot: robot1")
    print()
    print("🎭 Movement Pattern:")
    print("   Phase 1: Robot 1 moves RIGHT for 3 seconds")
    print("   Phase 2: Robot 1 moves LEFT for 3 seconds")
    print("   Final: Automatic alignment to vertical line")
    print()
    print("🤖 For Robot 2 (create separate file):")
    print("   Phase 1: Robot 2 moves LEFT for 3 seconds")
    print("   Phase 2: Robot 2 moves RIGHT for 3 seconds")
    print("   Final: Automatic alignment to vertical line")
    
    return filename

def create_robot2_choreography():
    """Create choreography for robot 2"""
    
    choreography_data = {
        "sample_advanced_move": {
            "type": "advanced_multi_phase",
            "primary_time": 3.0,
            "secondary_time": 3.0,
            "speed": 0.2,
            "timestamp": datetime.now().isoformat(),
            "robot_namespace": "robot2"
        }
    }
    
    filename = "choreography_robot2.json"
    with open(filename, 'w') as f:
        json.dump(choreography_data, f, indent=2)
    
    print(f"✅ Created robot 2 choreography: {filename}")
    return filename

def show_usage_instructions():
    """Show how to use the choreography system"""
    print("🚀 Usage Instructions:")
    print("=" * 50)
    print("1. Start the keyboard controller for each robot:")
    print("   python3 keyboard.py --robot robot1")
    print("   python3 keyboard.py --robot robot2")
    print()
    print("2. In each controller, press 'P' to play choreography")
    print("3. Select 'sample_advanced_move' from the list")
    print("4. Watch the synchronized movement!")
    print()
    print("🎯 Expected Result:")
    print("   - Both robots start at the same vertical line")
    print("   - Robot 1 moves right, Robot 2 moves left (3s)")
    print("   - Robot 1 moves left, Robot 2 moves right (3s)")
    print("   - Both robots end up aligned at the same vertical line")
    print()
    print("📊 Position Tracking:")
    print("   - Press 'i' to show current position")
    print("   - Press '0' to reset position to origin")
    print("   - Position is automatically tracked during movements")

if __name__ == "__main__":
    print("🎭 Advanced Multi-Phase Choreography Test")
    print("=" * 50)
    
    # Create sample choreographies
    robot1_file = create_sample_choreography()
    robot2_file = create_robot2_choreography()
    
    print()
    show_usage_instructions()
    
    print()
    print("📁 Created files:")
    print(f"   {robot1_file}")
    print(f"   {robot2_file}")
    print()
    print("🎬 Ready to test! Start the keyboard controllers and play the choreography.")
