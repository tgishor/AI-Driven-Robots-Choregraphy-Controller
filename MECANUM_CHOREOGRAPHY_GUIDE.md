# 🎭 Enhanced Mecanum Choreography System

## 🚀 New Capabilities Overview

The enhanced dance system now includes **16 distinct movement types** that leverage your Mecanum robot's unique capabilities, intelligently mapped to musical features.

## 🎵 Musical Feature Analysis

### Core Features Analyzed:
- **Energy**: RMS energy levels (loudness/intensity)
- **Brightness**: Spectral centroid (frequency content)
- **Onset Strength**: Attack characteristics (how sharp/sudden)
- **Duration**: Length of musical segment
- **Tempo**: Beat timing and rhythm

## 🤖 Movement Categories

### 🦾 **Servo-Based Movements** (Arm choreography)
*Perfect for melodic and expressive sections*

| Movement | Musical Trigger | Description |
|----------|----------------|-------------|
| `gentle_wave` | Low energy + soft tones | Soft, flowing arm waves |
| `energetic_wave` | High energy + bright | Sharp, dynamic arm movements |
| `deep_pulse` | Medium energy + bass | Rhythmic arm pulses |
| `bright_sparkle` | High brightness + quick | Fast, sparkling arm gestures |
| `flowing_reach` | Sustained + balanced | Graceful reaching movements |
| `dramatic_sweep` | High energy peaks | Bold, sweeping arm motions |
| `subtle_sway` | Gentle + bright | Light swaying movements |
| `powerful_strike` | Peak energy moments | Strong, accented arm strikes |

### 🛞 **Mecanum Base Movements** (Spatial choreography)
*Leveraging unique wheel capabilities*

| Movement | Musical Trigger | Description |
|----------|----------------|-------------|
| `sideways_slide` | Medium energy + smooth | Pure lateral movement (left/right) |
| `diagonal_drift` | Flowing + bright melodies | Forward-diagonal motion |
| `spin_in_place` | High energy + rotational feel | Pure rotation without translation |
| `circular_flow` | Sustained + circular patterns | Forward + rotation = circular path |
| `zigzag_dance` | Sharp + energetic | Quick directional changes |
| `smooth_glide` | Gentle + sustained | Forward gliding motion |
| `rhythmic_steps` | Rhythmic + bass-heavy | Rhythmic forward/backward steps |
| `explosive_burst` | Peak energy + sudden | Quick directional bursts |

## 🎼 Musical Feature → Movement Mapping

### **🔥 High Energy Sections**
```
Energy > 1.8 + Bright > 1.5 + Sharp Onset → explosive_burst
Energy > 1.5 + Bright > 1.5 + Strong Onset → spin_in_place  
Energy > 1.2 + Bright > 1.2 + Long Duration → zigzag_dance
```

### **🌊 Flowing Melodic Sections**
```
Bright > 1.3 + Medium Energy + Long Duration → sideways_slide
Energy > 1.0 + Bright > 1.2 + Sustained → diagonal_drift
Medium Energy + Sustained → circular_flow
```

### **🥁 Rhythmic/Bass Sections**
```
Energy > 1.0 + Low Brightness + Strong Onset → rhythmic_steps
Medium Energy + Low Brightness → deep_pulse
```

### **😌 Gentle/Ambient Sections**
```
Low Energy + Low Brightness → gentle_wave
Bright + Long Duration → smooth_glide
Low Energy + Medium Brightness → subtle_sway
```

## 🎯 Smart Movement Selection Logic

The system analyzes each beat segment and determines:

1. **Primary Movement Type**: Based on energy + brightness + onset
2. **Movement Category**: Servo-focused vs Base-focused
3. **Complementary Actions**: 
   - Base movements get subtle servo complements
   - Servo movements may trigger occasional base motion

## 🛡️ Safety & Precision Features

### **Bulletproof Execution**
- ✅ **2-second buffer** prevents mid-performance stops
- ✅ **Pre-calculated movements** eliminate real-time processing
- ✅ **Microsecond timing** using `time.perf_counter()`
- ✅ **Instant emergency stop** via `/dance/stop_command`

### **Movement Constraints**
- Servo positions: 150-850 pulse range (safety limits)
- Base speeds: Max 0.3 m/s linear, 1.5 rad/s angular
- Smooth transitions between movement types
- Graceful return to home position

## 🎮 Usage Examples

### **Start Performance**
```bash
# Auto-start with specific audio
python dance_attempt.py --audio /path/to/song.mp3 --start

# Manual start (wait for trigger)
python dance_attempt.py --audio /path/to/song.mp3
rostopic pub /dance/start_command std_msgs/Bool "data: true"
```

### **Emergency Stop**
```bash
python emergency_stop.py
# OR
rostopic pub /dance/stop_command std_msgs/Bool "data: true"
```

### **Demo Mecanum Capabilities**
```bash
python mecanum_dance_demo.py
```

## 🎵 Musical Examples

### **Electronic Dance Music (EDM)**
- Drop sections → `explosive_burst` + `spin_in_place`
- Build-ups → `zigzag_dance` + `energetic_wave`
- Breakdowns → `sideways_slide` + `flowing_reach`

### **Classical Music**
- String sections → `diagonal_drift` + `gentle_wave`
- Brass fanfares → `dramatic_sweep` + `circular_flow`
- Quiet passages → `smooth_glide` + `subtle_sway`

### **Hip-Hop/Rap**
- Heavy bass → `rhythmic_steps` + `deep_pulse`
- Vocal emphasis → `powerful_strike` + `explosive_burst`
- Flow sections → `sideways_slide` + `flowing_reach`

## 🔧 Customization

### **Adjust Movement Sensitivity**
Edit thresholds in `classify_movement_type()`:
```python
# Make spins more likely
elif energy > 1.2 and brightness > 1.0:  # Lower threshold
    return 'spin_in_place'
```

### **Add New Movements**
1. Add to `movement_types` dictionary
2. Implement in `calculate_base_movement()` or `calculate_servo_positions()`
3. Update classification logic

### **Modify Musical Analysis**
Adjust feature extraction in `analyze_complete_song()` for different music styles.

## 🎭 Performance Tips

1. **Song Selection**: Choose music with clear energy variations for best results
2. **Space Requirements**: Ensure adequate space for base movements
3. **Audio Quality**: Higher quality audio = better feature extraction
4. **Buffer Time**: Increase buffer for complex songs or slower systems
5. **Emergency Access**: Keep emergency stop command ready during performances

---

**The robot now truly dances with the music, combining expressive arm movements with dynamic spatial choreography!** 🎉🤖🎵
