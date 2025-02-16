audio_test.py

import sounddevice as sd
import pyaudio

def list_devices_sounddevice():
    """
    List devices using the sounddevice library.
    Shows name, max input/output channels, etc.
    """
    print("=== Sounddevice Device List ===")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        print(f"Index {idx}: {dev['name']}")
        print(f"   Max input channels:  {dev['max_input_channels']}")
        print(f"   Max output channels: {dev['max_output_channels']}")
    print("================================\n")


def list_devices_pyaudio():
    """
    List devices using the PyAudio library.
    Shows name and maxInputChannels for each device.
    """
    print("=== PyAudio Device List ===")
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    for i in range(device_count):
        info = p.get_device_info_by_index(i)
        print(f"Index {i}: {info['name']} (maxInputChannels={info['maxInputChannels']})")
    p.terminate()
    print("================================\n")


def open_pyaudio_stream(device_index=0, channels=1, rate=44100):
    """
    Open a PyAudio input stream on a specified device_index with given channels/rate.
    This is just a test example, does not record or save audio.
    """
    p = pyaudio.PyAudio()

    try:
        print(f"Opening stream on device index={device_index}, channels={channels}, rate={rate}")
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )
        print("Stream opened successfully!")
    except Exception as e:
        print(f"Could not open stream: {e}")
    finally:
        # Cleanup
        p.terminate()


def main():

    list_devices_sounddevice()

  
    list_devices_pyaudio()

    
    open_pyaudio_stream(device_index=device_index_to_test, channels=1, rate=44100)


if __name__ == "__main__":
    main()


tas install ka 
pip install sounddevice pyaudio


tas run mo sa script
python audio_test.py
