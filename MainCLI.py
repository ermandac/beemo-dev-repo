import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import sounddevice as sd
from BNBpredictor import predict_and_display, collect_new_data_and_labels as collect_bnb, retrain_model as retrain_bnb_model, save_results_to_google_sheets
from QNQpredictor import QNQpredictor, collect_new_data_and_labels as collect_qnq, retrain_qnq_model, handle_feedback, ask_true_label, load_and_preprocess_image, QNQ_save_results_to_google_sheets
from audio import list_audio_devices, select_audio_device, record_audio, audio_to_spectrogram, rename_file_based_on_content, get_next_filename
from Ctimer import start_scheduled_task
from datetime import time, datetime, timedelta
import asyncio
from discordbot import send_discord_message, run_discord_bot, bot, CHANNEL_ID
import logging
from tensorflow.keras.models import load_model
import pyaudio
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from scipy import signal
import json
from blynkout import send_string_to_blynk, run_blynk
from frequency import record_and_analyze, terminate_pyaudio
from loading import show_loading_screen
import schedule
import time as time_module
from gdrive import authenticate_drive, upload_to_drive
import glob

show_loading_screen()

# Configure logging for main.py
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main")

# Load models
base_dir = os.path.dirname(os.path.abspath(__file__))
bnb_model_path = os.path.join(base_dir, 'Model', 'updated_model_overall1.keras')
qnq_model_path = os.path.join(base_dir, 'Model', 'QNQupdated_model_overall17.keras')
bnb_model = load_model(bnb_model_path)
qnq_model = load_model(qnq_model_path)

# Global variables
START_TIME = None
END_TIME = None
DURATION_DAYS = None
INTERVAL = None

# Create a thread-safe queue for updates
update_queue = queue.Queue()

# Create a thread pool for background tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# Ensure spectrogram_filenames is defined
spectrogram_filenames = []

selected_device = None 
audio_api = None

def safe_matplotlib_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in matplotlib operation: {e}")
            return None
    return wrapper

@safe_matplotlib_operation
def process_audio_and_generate_spectrogram(audio_path, output_path):
    return audio_to_spectrogram(audio_path, output_path)

# Global variables
save_folder = os.path.join(base_dir, 'SavedIMG')
last_predicted_spectrogram = None
update_queue = queue.Queue()
thread_pool = ThreadPoolExecutor(max_workers=4)

async def record_and_spectrogram(channels, api):
    global last_predicted_spectrogram
    try:
        duration = 10
        audio_path = os.path.join(save_folder, 'recorded_audio.wav')
        temp_spectrogram_path = os.path.join(save_folder, 'temp_spectrogram.png')

        # Record audio
        await asyncio.to_thread(record_audio, audio_path, duration, channels, api)

        # Generate spectrogram
        final_spectrogram_path = await asyncio.to_thread(audio_to_spectrogram, audio_path, temp_spectrogram_path)

        if final_spectrogram_path:
            new_path = await asyncio.to_thread(rename_file_based_on_content, final_spectrogram_path, save_folder)
            update_queue.put(('display_spectrogram', new_path))
            update_queue.put(('update_filename_label', os.path.basename(new_path)))
            spectrogram_filenames.append(new_path) 
            update_queue.put(('update_saved_spectrograms_list',))
            last_predicted_spectrogram = new_path

            def get_selected_audio_device():
                devices = sd.query_devices()
                default_device = sd.default.device
                selected_device = devices[default_device[0]]  # Assuming the default input device is selected
                return selected_device['name'], selected_device['hostapi']
        
            # Perform predictions
            bnb_result = await asyncio.to_thread(predict_and_display, new_path)
            logger.debug(f"BNB Result: {bnb_result}")

            qnq_result = await asyncio.to_thread(QNQpredictor, new_path)
            logger.debug(f"QNQ Result: {qnq_result}")

            def convert_to_float(data): 
                if isinstance(data, np.float32): 
                    return float(data) 
                elif isinstance(data, list): 
                    return [convert_to_float(i) for i in data]
                elif isinstance(data, dict): 
                    return {k: convert_to_float(v) for k, v in data.items()} 
                return data 
            
            bnb_result = convert_to_float(bnb_result) 
            qnq_result = convert_to_float(qnq_result) 
            logger.debug(f"Converted BNB Result: {bnb_result}") 
            logger.debug(f"Converted QNQ Result: {qnq_result}")

            # Prepare the discord message
            discord_message = {
                "BNB Prediction": {
                    "File": os.path.basename(new_path),
                    "Predicted": "Bees Detected" if bnb_result[0] == 1 else "No Bees Detected",  # Readable prediction
                    "Confidence": float(bnb_result[1]) * 100,  # Ensure this is a float

                },
                "QNQ Prediction": {
                    "File": os.path.basename(new_path),
                    "Predicted": "Queen Detected" if qnq_result[0] == 1 else "No Queen Detected",  # Readable prediction
                    "Confidence": float(qnq_result[1]) * 100,  # Ensure this is a float

                }
            }

            # Send the message to Discord
            logger.debug(f"Sending to Discord: {json.dumps(discord_message, indent=2)}")
            await send_discord_message(json.dumps(discord_message, indent=2), new_path)

            # Send the predictions to Blynk
            bnb_prediction = "Bees Detected" if bnb_result[0] == 1 else "No Bees Detected"
            qnq_prediction = "Queen Detected" if qnq_result[0] == 1 else "No Queen Detected"
            send_string_to_blynk(2, bnb_prediction)  # V2 for BNB
            send_string_to_blynk(3, qnq_prediction)  # V3 for QNQ

            selected_device = get_selected_audio_device()
            if selected_device:
                audio_device_message = f"Selected device: {selected_device} with API"
                logger.info(audio_device_message)
                await send_discord_message(audio_device_message, new_path)

            
    except Exception as e:
        logger.error(f"An error occurred during recording and spectrogram generation: {e}")

async def record_and_spectrogram_worker(device_info):
    channels, api = device_info

    # Define PyAudio instance and audio format parameters
    p = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    RATE = 44100
    CHUNK = 1024

    # Open stream
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    await record_and_spectrogram(channels, api)

    # Call the function to record and analyze audio for 15 seconds
    await asyncio.create_task(record_and_analyze(bot, CHANNEL_ID, stream, duration=15))
    
    # Terminate PyAudio
    terminate_pyaudio()
    
    await record_and_spectrogram(channels, api)

    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

async def notify_discord(status):
    await send_discord_message(f"Microphone {status}", None)

def display_spectrogram(image_path):
    update_queue.put(('display_spectrogram', image_path))

async def run_scheduled_task(bot, channel_id):
    print("Running scheduled task...")
    # Define PyAudio instance and audio format parameters
    p = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    RATE = 44100
    CHANNELS = 1
    CHUNK = 1024

    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Call the function to record and analyze audio for 10 seconds
    await record_and_analyze(bot, channel_id, stream, duration=10)
    
    # Perform BNB prediction
    bnb_results = predict_and_display()
    save_results_to_google_sheets(bnb_results)

    # Perform QNQ prediction
    qnq_results = QNQpredictor()
    QNQ_save_results_to_google_sheets(qnq_results)

    # Send results to Discord
    message = f"BNB Results: {bnb_results}\nQNQ Results: {qnq_results}"
    await send_discord_message(bot, channel_id, message)
    
    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

async def start_discord_bot():
    await run_discord_bot()

def is_microphone_connected():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    for i in range(0, num_devices):
        if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
            p.terminate()
            return True
    p.terminate()
    return False

async def start_recording():
    if is_microphone_connected():
        await notify_discord("connected")
        
        device_info = (selected_device, audio_api)
        await record_and_spectrogram_worker(device_info)
        
        await notify_discord("processing complete")
    else:
        await notify_discord("disconnected")

def trigger_recording():
    loop = asyncio.get_event_loop()
    loop.create_task(start_recording())

def start_scheduled_task(bot, channel_id):
    # Schedule the task to run at the specified interval
    schedule.every(INTERVAL).seconds.do(lambda: asyncio.run(run_scheduled_task(bot, channel_id)))

    # Run the scheduler
    def run_scheduler():
        while True:
            current_time = datetime.now().time()
            if START_TIME <= current_time <= END_TIME:
                schedule.run_pending()
            time_module.sleep(1)

    # Start the scheduler in a separate thread
    threading.Thread(target=run_scheduler, daemon=True).start()

def main():
    print("Welcome to the CLI version of the application.")
    print("Please select an audio device:")
    devices = list_audio_devices()
    for i, device in enumerate(devices):
        print(f"{i + 1}. {device['name']}")
    device_index = int(input("Enter the device number: ")) - 1
    selected_device, audio_api = select_audio_device(device_index)
    print(f"Selected device: {selected_device} with API {audio_api}")

    loop = asyncio.get_event_loop()
    
    discord_thread = threading.Thread(target=lambda: loop.run_until_complete(start_discord_bot()))
    discord_thread.start()

    while True:
        print("\nOptions:")
        print("1. Start Recording")
        print("2. Set Schedule")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            trigger_recording()
        elif choice == "2":
            start_hour = int(input("Enter start hour (0-23): "))
            start_minute = int(input("Enter start minute (0-59): "))
            end_hour = int(input("Enter end hour (0-23): "))
            end_minute = int(input("Enter end minute (0-59): "))
            duration_days = int(input("Enter duration in days: "))
            interval = int(input("Enter interval in seconds: "))

            global START_TIME, END_TIME, DURATION_DAYS, INTERVAL
            START_TIME = time(start_hour, start_minute)
            END_TIME = time(end_hour, end_minute)
            DURATION_DAYS = duration_days
            INTERVAL = interval

            start_scheduled_task(bot, CHANNEL_ID)
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()