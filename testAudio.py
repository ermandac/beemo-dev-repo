import pyaudio

# Initialize PyAudio
p = pyaudio.PyAudio()

print("Available audio devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    # Check if the device supports input
    if info['maxInputChannels'] > 0:
        print(f"Device ID {i}: {info['name']} (Input Channels: {info['maxInputChannels']})")

# Terminate PyAudio
p.terminate()