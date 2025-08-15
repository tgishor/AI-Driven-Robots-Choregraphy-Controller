from pydub import AudioSegment

# Load your audio file
song = AudioSegment.from_file("E:\OpenDayRobotics\AI-Driven-Robots-Choregraphy-Controller\src\jetrover_controller\jetrover_controller\On-The-Floor.mp3", format="mp3")

# Set trim points (i    n milliseconds)
start_time = 30 * 1000   # 30 seconds
end_time = 60 * 1000     # 60 seconds

# Trim audio
trimmed_song = song[start_time:end_time]

# Export trimmed audio
trimmed_song.export("trimmed_song.mp3", format="mp3")

print("Song trimmed successfully!")



## FUCK THIS IS NEW MF