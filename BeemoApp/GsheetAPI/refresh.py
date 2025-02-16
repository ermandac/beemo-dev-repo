import os
import json
import google.auth.transport.requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def get_creds():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth flow is completed to obtain new credentials.
    """
    creds = None
    token_path = 'token.json'
    creds_path = r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\BeemoApp\GsheetAPI\BNB_new_client_secret.json'  # Ensure correct path

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    return creds

# Use get_creds to obtain credentials
creds = get_creds()

# Now you can use these credentials to access Google Sheets API














import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
import tkinter.simpledialog
from PIL import Image, ImageTk, ImageSequence
import sounddevice as sd
from tkinter import filedialog
from BNBpredictor import predict_and_display, collect_new_data_and_labels as collect_bnb, retrain_model as retrain_bnb_model, save_results_to_google_sheets
from QNQpredictor import QNQpredictor, collect_new_data_and_labels as collect_qnq, retrain_qnq_model, handle_feedback, ask_true_label, load_and_preprocess_image, QNQ_save_results_to_google_sheets
from audio import list_audio_devices, select_audio_device, record_audio, audio_to_spectrogram, rename_file_based_on_content, get_next_filename
from Ctimer import start_scheduled_task
from datetime import time, datetime
import asyncio
from discordbot import send_discord_message, run_discord_bot
import logging
from tensorflow.keras.models import load_model
import pyaudio
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from scipy import signal
import  json


# Configure logging for main.py
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main")

# Load models
bnb_model_path = r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\Model\updated_model_overall1.keras'
qnq_model_path = r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\Model\QNQupdated_model_overall17.keras'
bnb_model = load_model(bnb_model_path)
qnq_model = load_model(qnq_model_path)

# Initialize event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Create a thread-safe queue for GUI updates
gui_update_queue = queue.Queue()

# Create a thread pool for background tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# Ensure spectrogram_filenames is defined
spectrogram_filenames = []

def safe_matplotlib_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in matplotlib operation: {e}")
            return None
    return wrapper

@safe_matplotlib_operation
def process_audio_and_generate_spectrogram(audio_path, output_path):
    # This function will be called in a background thread
    return audio_to_spectrogram(audio_path, output_path)

async def record_and_spectrogram(channels, api):
    global last_predicted_spectrogram
    try:
        duration = 10
        audio_path = os.path.join(save_folder, 'recorded_audio.wav')
        temp_spectrogram_path = os.path.join(save_folder, 'temp_spectrogram.png')
        await asyncio.to_thread(record_audio, audio_path, duration, channels, api)
        final_spectrogram_path = await asyncio.to_thread(audio_to_spectrogram, audio_path, temp_spectrogram_path)
        if final_spectrogram_path:
            new_path = await asyncio.to_thread(rename_file_based_on_content, final_spectrogram_path, save_folder)
            gui_update_queue.put(('display_spectrogram', new_path))
            gui_update_queue.put(('update_filename_label', os.path.basename(new_path)))
            spectrogram_filenames.append(new_path)
            gui_update_queue.put(('update_saved_spectrograms_list',))
            last_predicted_spectrogram = new_path

            output_box1.delete(1.0, tk.END)
            bnb_result = await asyncio.to_thread(predict_and_display, new_path, output_box1)
            print(f"Debug BNB Result: {bnb_result}")

            output_box2.delete(1.0, tk.END)
            qnq_result = await asyncio.to_thread(QNQpredictor, new_path, output_box2)
            print(f"Debug QNQ Result: {qnq_result}")

            def convert_to_float(data):
                """Recursively convert all numpy float32 to float"""
                if isinstance(data, np.float32):
                    return float(data)
                elif isinstance(data, list):
                    return [convert_to_float(i) for i in data]
                elif isinstance(data, dict):
                    return {k: convert_to_float(v) for k, v in data.items()}
                return data

            bnb_result = convert_to_float(bnb_result)
            qnq_result = convert_to_float(qnq_result)
            print(f"Converted BNB Result: {bnb_result}")
            print(f"Converted QNQ Result: {qnq_result}")

            discord_message = {
                "BNB Prediction": {
                    "File": os.path.basename(new_path),
                    "Predicted": bnb_result[0],
                    "Confidence": bnb_result[1] * 100,
                    "Accuracy": bnb_result[2] if len(bnb_result) > 2 else 'N/A',
                    "Precision": bnb_result[3] if len(bnb_result) > 3 else 'N/A',
                    "Recall": bnb_result[4] if len(bnb_result) > 4 else 'N/A',
                    "F1 Score": bnb_result[5] if len(bnb_result) > 5 else 'N/A',
                    "Model Retrained": bnb_result[6] if len(bnb_result) > 6 else 'N/A'
                },
                "QNQ Prediction": {
                    "File": os.path.basename(new_path),
                    "Predicted": qnq_result[0],
                    "Confidence": qnq_result[1] * 100,
                    "Accuracy": qnq_result[2] if len(qnq_result) > 2 else 'N/A',
                    "Precision": qnq_result[3] if len(qnq_result) > 3 else 'N/A',
                    "Recall": qnq_result[4] if len(qnq_result) > 4 else 'N/A',
                    "F1 Score": qnq_result[5] if len(qnq_result) > 5 else 'N/A',
                    "Model Retrained": qnq_result[6] if len(qnq_result) > 6 else 'N/A'
                }
            }

            print(f"Sending to Discord: {json.dumps(discord_message, indent=2)}")
            await send_discord_message(json.dumps(discord_message, indent=2), new_path)

    except Exception as e:
        print(f"An error occurred during recording and spectrogram generation: {e}")




def display_spectrogram(image_path):
    gui_update_queue.put(('display_spectrogram', image_path))

def open_schedule_settings():
    schedule_window = tk.Toplevel()
    schedule_window.title("Schedule Settings")
    schedule_window.geometry("400x350")
    schedule_window.configure(bg='#6B4A29')

    style = ttk.Style(schedule_window)
    style.theme_use('clam')
    style.configure("TLabel", background="#6B4A29", foreground="#F9D923", font=('Helvetica', 12, 'bold'))
    style.configure("TButton", background="#5C3A21", foreground="#F9D923", font=('Helvetica', 12, 'bold'))
    style.map("TButton", background=[('active', '#6D4E3D')])

    ttk.Label(schedule_window, text="Start Time (HH:MM):", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, padx=10, pady=10, sticky='w')
    start_hour_spinbox = ttk.Spinbox(schedule_window, from_=0, to=23, width=5, font=('Helvetica', 12))
    start_hour_spinbox.grid(row=0, column=1, padx=5, pady=10, sticky='w')
    start_minute_spinbox = ttk.Spinbox(schedule_window, from_=0, to=59, width=5, font=('Helvetica', 12))
    start_minute_spinbox.grid(row=0, column=2, padx=5, pady=10, sticky='w')

    ttk.Label(schedule_window, text="End Time (HH:MM):", font=('Helvetica', 12, 'bold')).grid(row=1, column=0, padx=10, pady=10, sticky='w')
    end_hour_spinbox = ttk.Spinbox(schedule_window, from_=0, to=23, width=5, font=('Helvetica', 12))
    end_hour_spinbox.grid(row=1, column=1, padx=5, pady=10, sticky='w')
    end_minute_spinbox = ttk.Spinbox(schedule_window, from_=0, to=59, width=5, font=('Helvetica', 12))
    end_minute_spinbox.grid(row=1, column=2, padx=5, pady=10, sticky='w')

    ttk.Label(schedule_window, text="Duration (days):", font=('Helvetica', 12, 'bold')).grid(row=2, column=0, padx=10, pady=10, sticky='w')
    duration_days_spinbox = ttk.Spinbox(schedule_window, from_=1, to=365, width=5, font=('Helvetica', 12))
    duration_days_spinbox.grid(row=2, column=1, padx=10, pady=10, sticky='w')

    ttk.Label(schedule_window, text="Interval (seconds):", font=('Helvetica', 12, 'bold')).grid(row=3, column=0, padx=10, pady=10, sticky='w')
    interval_spinbox = ttk.Spinbox(schedule_window, from_=10, to=86400, width=5, font=('Helvetica', 12))
    interval_spinbox.grid(row=3, column=1, padx=10, pady=10, sticky='w')

    ttk.Label(schedule_window, text="Activate Schedule:", font=('Helvetica', 12, 'bold')).grid(row=4, column=0, padx=10, pady=10, sticky='w')
    activate_var = tk.BooleanVar()
    activate_switch = ttk.Checkbutton(schedule_window, variable=activate_var)
    activate_switch.grid(row=4, column=1, padx=10, pady=10, sticky='w')

    def save_schedule():
        global START_TIME, END_TIME, DURATION_DAYS, INTERVAL
        start_hour = start_hour_spinbox.get()
        start_minute = start_minute_spinbox.get()
        end_hour = end_hour_spinbox.get()
        end_minute = end_minute_spinbox.get()
        duration_days = duration_days_spinbox.get()
        interval = interval_spinbox.get()

        START_TIME = time(int(start_hour), int(start_minute))
        END_TIME = time(int(end_hour), int(end_minute))
        DURATION_DAYS = int(duration_days)
        INTERVAL = int(interval)
        if activate_var.get():
            start_scheduled_task()

        schedule_window.destroy()

    save_button = ttk.Button(schedule_window, text="Save", command=save_schedule, style='TButton')
    save_button.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

    schedule_window.mainloop()

def create_gui():
    global output_box1, output_box2, filename_label, save_folder, saved_spectrogram_frame, saved_spectrograms_listbox, font, spectrogram_frame, root
    save_folder = r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\SavedIMG'
    saved_spectrogram_filenames = []  # Initialize the list here

    root = tk.Tk()
    root.title("BEEMO 2.0 - BEEMO AUDIO ANALYSIS")
    root.iconphoto(False, tk.PhotoImage(file=r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\Icon\icon3.png'))
    root.configure(bg='#6B4A29')

    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure("TLabel", background="#6B4A29", foreground="#F9D923", font=('Helvetica', 12, 'bold'))
    style.configure("TButton", background="#5C3A21", foreground="#F9D923", font=('Helvetica', 12, 'bold'))
    style.map("TButton", background=[('active', '#6D4E3D')])
    style.configure("TEntry", fieldbackground="#4D4D4D", foreground="#F9D923", font=('Helvetica', 12))
    style.configure("TOptionMenu", background="#5C3A21", foreground="#F9D923", font=('Helvetica', 12))
    style.configure("TFrame", background="#6B4A29")
    style.configure("TListbox", background="#4D4D4D", foreground="#F9D923", font=('Helvetica', 12))

    selected_device = tk.StringVar()
    spectrogram_filenames = []
    last_predicted_spectrogram = None

    def list_devices():
        devices = sd.query_devices()
        input_devices = [device for device in devices if device['max_input_channels'] > 0]
        return input_devices

    def populate_device_dropdown():
        input_devices = list_devices()
        device_names = [device['name'] for device in input_devices]
        menu = device_dropdown['menu']
        menu.delete(0, 'end')
        for device_name in device_names:
            menu.add_command(label=device_name, command=tk._setit(selected_device, device_name))
        if device_names:
            selected_device.set(device_names[0])

    def select_device():
        device_name = selected_device.get()
        input_devices = list_devices()
        for device in input_devices:
            if device['name'] == device_name:
                device_index = input_devices.index(device)
                channels, selected_device_info = select_audio_device(device_index)
                if selected_device_info:
                    api = selected_device_info.get('hostapi', 'Unknown')
                    print(f"Selected device: {device_name} with API {api}")
                    return channels, api
        return 1, None

    def on_select_device():
        threading.Thread(target=record_and_spectrogram_worker, args=(select_device(),)).start()

    def record_and_spectrogram_worker(device_info):
        channels, api = device_info
        asyncio.run(record_and_spectrogram(channels, api))

    async def record_and_spectrogram(channels, api):
        global last_predicted_spectrogram
        try:
            duration = 10
            audio_path = os.path.join(save_folder, 'recorded_audio.wav')
            temp_spectrogram_path = os.path.join(save_folder, 'temp_spectrogram.png')
            await asyncio.to_thread(record_audio, audio_path, duration, channels, api)
            final_spectrogram_path = await asyncio.to_thread(audio_to_spectrogram, audio_path, temp_spectrogram_path)
            if final_spectrogram_path:
                new_path = await asyncio.to_thread(rename_file_based_on_content, final_spectrogram_path, save_folder)
                gui_update_queue.put(('display_spectrogram', new_path))
                gui_update_queue.put(('update_filename_label', os.path.basename(new_path)))
                spectrogram_filenames.append(new_path)
                gui_update_queue.put(('update_saved_spectrograms_list',))
                last_predicted_spectrogram = new_path

                output_box1.delete(1.0, tk.END)
                bnb_result = await asyncio.to_thread(predict_and_display, new_path, output_box1)

                output_box2.delete(1.0, tk.END)
                qnq_result = await asyncio.to_thread(QNQpredictor, new_path, output_box2)

                bnb_prediction_result, bnb_confidence = map(float, bnb_result[:2])
                qnq_prediction_result, qnq_confidence = map(float, qnq_result[:2])

                discord_message = (
                    f"**BNB Prediction**:\n"
                    f"File: {os.path.basename(new_path)}\n"
                    f"Predicted: {bnb_prediction_result}\n"
                    f"Confidence: {bnb_confidence * 100:.2f}%\n"
                    f"Accuracy: {float(bnb_result[2]) if len(bnb_result) > 2 else 'N/A'}\n"
                    f"Precision: {float(bnb_result[3]) if len(bnb_result) > 3 else 'N/A'}\n"
                    f"Recall: {float(bnb_result[4]) if len(bnb_result) > 4 else 'N/A'}\n"
                    f"F1 Score: {float(bnb_result[5]) if len(bnb_result) > 5 else 'N/A'}\n"
                    f"Model Retrained: {bnb_result[6] if len(bnb_result) > 6 else 'N/A'}\n\n"
                    f"**QNQ Prediction**:\n"
                    f"File: {os.path.basename(new_path)}\n"
                    f"Predicted: {qnq_prediction_result}\n"
                    f"Confidence: {qnq_confidence * 100:.2f}%\n"
                    f"Accuracy: {float(qnq_result[2]) if len(qnq_result) > 2 else 'N/A'}\n"
                    f"Precision: {float(qnq_result[3]) if len(qnq_result) > 3 else 'N/A'}\n"
                    f"Recall: {float(qnq_result[4]) if len(qnq_result) > 4 else 'N/A'}\n"
                    f"F1 Score: {float(qnq_result[5]) if len(qnq_result) > 5 else 'N/A'}\n"
                    f"Model Retrained: {qnq_result[6] if len(qnq_result) > 6 else 'N/A'}\n"
                )
                print(f"Sending to Discord: {discord_message}")

                await send_discord_message(discord_message, new_path)

        except Exception as e:
            print(f"An error occurred during recording and spectrogram generation: {e}")

    def handle_display_spectrogram(image_path):
        global spectrogram_label
        try:
            for widget in spectrogram_frame.winfo_children():
                widget.destroy()

            img = Image.open(image_path)
            img = img.resize((400, 300), Image.LANCZOS)  # Adjust size for better centering
            img = ImageTk.PhotoImage(img)
            spectrogram_label = ttk.Label(spectrogram_frame, image=img)
            spectrogram_label.image = img
            spectrogram_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Center the spectrogram

        except Exception as e:
            print(f"Error displaying spectrogram: {e}")

    def update_saved_spectrograms_list():
        saved_spectrograms_listbox.delete(0, tk.END)
        for filename in os.listdir(save_folder):
            if filename.endswith('.png') and filename.startswith("BeemoDosSpectrogram_"):
                saved_spectrograms_listbox.insert(tk.END, filename)
                saved_spectrogram_filenames.append(os.path.join(save_folder, filename))

    def on_spectrogram_select(event):
        selected_index = saved_spectrograms_listbox.curselection()
        if selected_index:
            selected_filename = saved_spectrograms_listbox.get(selected_index)
            selected_path = os.path.join(save_folder, selected_filename)
            filename_label.configure(text=selected_filename, anchor='center')
            display_spectrogram(selected_path)
            if selected_path != last_predicted_spectrogram:
                output_box1.delete(1.0, tk.END)
                predict_and_display(selected_path, output_box1)

                output_box2.delete(1.0, tk.END)
                QNQpredictor(selected_path, output_box2)
                last_predicted_spectrogram = selected_path

    def set_save_folder():
        global save_folder
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            save_folder = folder_selected
            update_saved_spectrograms_list()

    def on_resize(event):
        new_size = max(10, int(root.winfo_width() / 100))
        font.configure(size=new_size)
        saved_spectrograms_listbox.configure(font=font)
        saved_spectrograms_label.configure(font=font)
        title_label.configure(font=font)
        filename_label.configure(font=font)

    # Function to find the latest file in a directory
    def find_latest_file(directory):
        files = os.listdir(directory)
        paths = [os.path.join(directory, basename) for basename in files]
        latest_file = max(paths, key=os.path.getctime)
        return latest_file

    # Function to handle QNQ commands
    def on_qnq_command(event=None):
        command = qnq_command_entry.get().strip().lower()
        if command == "correct":
            qnq_feedback_label.configure(text="Feedback received: QNQ Prediction was correct.")
        elif command == "incorrect":
            qnq_feedback_label.configure(text="Feedback received: QNQ Prediction was incorrect. Retraining...")
            img_path = find_latest_file(r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\SavedIMG')
            true_label = ask_true_label("Enter the true label (0 for No Queen Detected, 1 for Queen Detected):")
            new_data, new_labels = collect_qnq(true_label, img_path)
            retrain_qnq_model(qnq_model, new_data, new_labels)
            predicted_class, confidence, f1, precision = QNQpredictor(img_path, output_box2)
            QNQ_save_results_to_google_sheets(img_path, true_label, predicted_class, confidence, f1, precision, model_retrained=True)
            qnq_feedback_label.configure(text="QNQ Model retrained successfully.")
        else:
            qnq_feedback_label.configure(text="Invalid command. Please type 'correct' or 'incorrect'.")
        qnq_command_entry.delete(0, tk.END)

    def on_command(event=None):
        command = command_entry.get().strip().lower()
        if command == "correct":
            feedback_label.configure(text="Feedback received: Prediction was correct.")
        elif command == "incorrect":
            feedback_label.configure(text="Feedback received: Prediction was incorrect. Retraining...")
            img_path = find_latest_file(r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\SavedIMG')
            true_label = ask_true_label("Enter the true label (0 for No Bees, 1 for Bees):")
            new_data, new_labels = collect_bnb(true_label, img_path)
            retrain_bnb_model(bnb_model, new_data, new_labels)
            predicted_class, confidence, f1, precision = predict_and_display(img_path, output_box1)
            save_results_to_google_sheets(img_path, true_label, predicted_class, confidence, f1, precision, model_retrained=True)
            feedback_label.configure(text="BNB Model retrained successfully.")
        else:
            feedback_label.configure(text="Invalid command. Please type 'correct' or 'incorrect'.")
        command_entry.delete(0, tk.END)

    # Function to ask for manual input of the true label
    def ask_true_label(prompt):
        true_label = tk.simpledialog.askinteger("Input", prompt)
        return true_label

    def toggle_saved_spectrograms():
        if saved_spectrogram_frame.winfo_ismapped():
            saved_spectrogram_frame.grid_remove()
        else:
            saved_spectrogram_frame.grid()

    font = Font(family="Helvetica", size=12)
    title_font = Font(family="Helvetica", size=17, weight="bold")

    canvas = tk.Canvas(root, bg='#6B4A29')
    scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scroll_x = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    frame = ttk.Frame(canvas, padding="10")
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    top_frame = ttk.Frame(frame, padding="10")
    top_frame.grid(row=0, column=0, columnspan=4, sticky="ew")

    title_label = ttk.Label(top_frame, text="BEEMO AUDIO ANALYSIS", font=title_font)
    title_label.grid(column=0, row=0, padx=10, pady=10, sticky="w")

    beemo_path = r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\Icon\BeemoDos.png'
    beemo_image = Image.open(beemo_path).resize((40, 40), Image.LANCZOS)
    beemo_photo = ImageTk.PhotoImage(beemo_image)
    beemo_button = tk.Button(top_frame, image=beemo_photo, bg='#6B4A29', activebackground='#6B4A29', borderwidth=0)
    beemo_button.image = beemo_photo
    beemo_button.grid(column=3, row=0, padx=10, pady=10, sticky="e")

    device_dropdown_label = ttk.Label(top_frame, text="Select Device:", font=font)
    device_dropdown_label.grid(column=0, row=1, padx=10, pady=10, sticky="w")

    device_dropdown = ttk.OptionMenu(top_frame, selected_device, ())
    device_dropdown.grid(column=1, row=1, padx=10, pady=10, sticky="w")

    show_device_button = ttk.Button(top_frame, text="Show Devices", command=populate_device_dropdown, style='TButton')
    show_device_button.grid(column=2, row=1, padx=10, pady=10)

    select_button = ttk.Button(top_frame, text="Select Device and Record", command=on_select_device, style='TButton')
    select_button.grid(column=3, row=1, padx=10, pady=10)

    save_folder_button = ttk.Button(top_frame, text="Save Folder", command=set_save_folder, style='TButton')
    save_folder_button.grid(column=0, row=2, columnspan=4, padx=10, pady=10, sticky="ew")

    filename_label = ttk.Label(top_frame, text="", font=font, anchor='center')
    filename_label.grid(column=0, row=3, columnspan=4, padx=10, pady=10, sticky="ew")

    middle_frame = ttk.Frame(frame, padding="10")
    middle_frame.grid(row=1, column=0, sticky="nsew")
    middle_frame.columnconfigure(0, weight=1)
    middle_frame.rowconfigure(0, weight=8)
    middle_frame.rowconfigure(1, weight=1)

    spectrogram_frame = ttk.Frame(middle_frame)
    spectrogram_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    spectrogram_frame.columnconfigure(0, weight=1)
    spectrogram_frame.rowconfigure(0, weight=1)

    output_frame = ttk.Frame(middle_frame)
    output_frame.grid(row=1, column=0, pady=10, padx=10, sticky="ew")
    output_box1 = tk.Text(output_frame, height=10, width=50, bg="#4D4D4D", fg="#FFFFFF", font=font)
    output_box1.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    output_box2 = tk.Text(output_frame, height=10, width=50, bg="#4D4D4D", fg="#FFFFFF", font=font)
    output_box2.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    command_frame = ttk.Frame(frame, padding="10")
    command_frame.grid(row=2, column=0, sticky="ew")
    command_label = ttk.Label(command_frame, text="Enter BNB Command (correct/incorrect):", font=font)
    command_label.grid(column=0, row=0, padx=10, pady=5, sticky="w")
    command_entry = ttk.Entry(command_frame, font=font)
    command_entry.grid(column=1, row=0, padx=10, pady=5, sticky="ew")
    command_entry.bind("<Return>", on_command)
    command_button = ttk.Button(command_frame, text="Submit", command=on_command, style='TButton')
    command_button.grid(column=2, row=0, padx=10, pady=5)

    feedback_label = ttk.Label(command_frame, text="", font=font)
    feedback_label.grid(column=0, row=1, columnspan=3, padx=10, pady=5, sticky="ew")

    qnq_command_frame = ttk.Frame(frame, padding="10")
    qnq_command_frame.grid(row=3, column=0, sticky="ew")
    qnq_command_label = ttk.Label(qnq_command_frame, text="Enter QNQ Command (correct/incorrect):", font=font)
    qnq_command_label.grid(column=0, row=0, padx=10, pady=5, sticky="w")
    qnq_command_entry = ttk.Entry(qnq_command_frame, font=font)
    qnq_command_entry.grid(column=1, row=0, padx=10, pady=5, sticky="ew")
    qnq_command_entry.bind("<Return>", on_qnq_command)
    qnq_command_button = ttk.Button(qnq_command_frame, text="Submit", command=on_qnq_command, style='TButton')
    qnq_command_button.grid(column=2, row=0, padx=10, pady=5)

    qnq_feedback_label = ttk.Label(qnq_command_frame, text="", font=font)
    qnq_feedback_label.grid(column=0, row=1, columnspan=3, padx=10, pady=5, sticky="ew")

    saved_images_path = r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\Icon\SavedSpectro.png'
    saved_images_image = Image.open(saved_images_path).resize((40, 40), Image.LANCZOS)
    saved_images_photo = ImageTk.PhotoImage(saved_images_image)
    saved_images_button = tk.Button(root, image=saved_images_photo, command=toggle_saved_spectrograms, bg='#6B4A29', activebackground='#6B4A29', borderwidth=0)
    saved_images_button.image = saved_images_photo
    saved_images_button.place(relx=0.5, rely=0.98, anchor='s')

    saved_spectrogram_frame = ttk.Frame(frame, padding="10", style='TFrame')
    saved_spectrogram_frame.grid(row=4, column=0, sticky="nsew")
    saved_spectrogram_frame.columnconfigure(0, weight=1)
    saved_spectrogram_frame.rowconfigure(1, weight=1)

    saved_spectrograms_label = ttk.Label(saved_spectrogram_frame, text="Saved Images", font=font)
    saved_spectrograms_label.grid(column=0, row=0, padx=10, pady=10)

    saved_spectrograms_listbox = tk.Listbox(saved_spectrogram_frame, height=20, bg="#4D4D4D", fg="#F9D923", font=font)
    saved_spectrograms_listbox.grid(column=0, row=1, padx=10, pady=10, sticky="nsew")
    saved_spectrograms_listbox.bind('<<ListboxSelect>>', on_spectrogram_select)

    update_saved_spectrograms_list()

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=1)
    root.rowconfigure(3, weight=1)
    root.rowconfigure(4, weight=1)

    top_frame.columnconfigure(0, weight=1)
    top_frame.columnconfigure(1, weight=1)
    top_frame.columnconfigure(2, weight=1)
    top_frame.columnconfigure(3, weight=1)

    middle_frame.columnconfigure(0, weight=1)
    middle_frame.rowconfigure(0, weight=8)
    middle_frame.rowconfigure(1, weight=1)

    command_frame.columnconfigure(0, weight=1)
    command_frame.columnconfigure(1, weight=1)
    command_frame.columnconfigure(2, weight=1)

    qnq_command_frame.columnconfigure(0, weight=1)
    qnq_command_frame.columnconfigure(1, weight=1)
    qnq_command_frame.columnconfigure(2, weight=1)

    saved_spectrogram_frame.grid_remove()

    clock_path = r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\Icon\clock.png'
    clock_image = Image.open(clock_path).convert("RGBA")
    clock_image = clock_image.resize((30, 30), Image.LANCZOS)
    clock_photo = ImageTk.PhotoImage(clock_image)

    schedule_button = tk.Button(root, image=clock_photo, command=open_schedule_settings, bg='#6B4A29', activebackground='#6B4A29', borderwidth=0)
    schedule_button.image = clock_photo
    schedule_button.place(relx=0.95, rely=0.95, anchor="se")

    def process_tasks():
        while not gui_update_queue.empty():
            try:
                task = gui_update_queue.get_nowait()
                if task[0] == 'display_spectrogram':
                    handle_display_spectrogram(task[1])
                elif task[0] == 'update_filename_label':
                    filename_label.configure(text=task[1], anchor='center')
                elif task[0] == 'update_saved_spectrograms_list':
                    update_saved_spectrograms_list()
            except queue.Empty:
                pass
        root.after(100, process_tasks)  # Continue processing tasks every 100ms

    root.after(100, process_tasks)  # Start processing the task queue

    root.mainloop()

def start_discord_bot():
    def run():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        new_loop.run_until_complete(run_discord_bot())

    threading.Thread(target=run).start()

# Function to check if microphone is connected
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

# Function to notify Discord about microphone status
async def notify_discord(status):
    await send_discord_message(f"Microphone {status}", None)

# Function to start recording process and send data
async def start_recording():
    if is_microphone_connected():
        await notify_discord("connected")
        # Implement the recording and data sending logic here
    else:
        await notify_discord("disconnected")

def trigger_recording():
    asyncio.run(start_recording())

if __name__ == "__main__":
    start_discord_bot()  # Run the Discord bot in a separate thread
    create_gui()  # Create the GUI















    ####################################################









import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
import tkinter.simpledialog
from PIL import Image, ImageTk, ImageSequence
import sounddevice as sd
from tkinter import filedialog
from BNBpredictor import predict_and_display, collect_new_data_and_labels as collect_bnb, retrain_model as retrain_bnb_model, save_results_to_google_sheets
from QNQpredictor import QNQpredictor, collect_new_data_and_labels as collect_qnq, retrain_qnq_model, handle_feedback, ask_true_label, load_and_preprocess_image, QNQ_save_results_to_google_sheets
from audio import list_audio_devices, select_audio_device, record_audio, audio_to_spectrogram, rename_file_based_on_content, get_next_filename
from Ctimer import start_scheduled_task
from datetime import time, datetime
import asyncio
from discordbot import send_discord_message, run_discord_bot
import logging
from tensorflow.keras.models import load_model
import pyaudio
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from scipy import signal
import json

# Configure logging for main.py
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main")

# Load models
bnb_model_path = r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\Model\updated_model_overall1.keras'
qnq_model_path = r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\Model\QNQupdated_model_overall17.keras'
bnb_model = load_model(bnb_model_path)
qnq_model = load_model(qnq_model_path)

# Initialize event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Create a thread-safe queue for GUI updates
gui_update_queue = queue.Queue()

# Create a thread pool for background tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# Ensure spectrogram_filenames is defined
spectrogram_filenames = []

def safe_matplotlib_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in matplotlib operation: {e}")
            return None
    return wrapper

@safe_matplotlib_operation
def process_audio_and_generate_spectrogram(audio_path, output_path):
    # This function will be called in a background thread
    return audio_to_spectrogram(audio_path, output_path)

# Simplified and Debugged record_and_spectrogram
async def record_and_spectrogram(channels, api):
    global last_predicted_spectrogram
    try:
        duration = 10
        audio_path = os.path.join(save_folder, 'recorded_audio.wav')
        temp_spectrogram_path = os.path.join(save_folder, 'temp_spectrogram.png')
        await asyncio.to_thread(record_audio, audio_path, duration, channels, api)
        final_spectrogram_path = await asyncio.to_thread(audio_to_spectrogram, audio_path, temp_spectrogram_path)
        if final_spectrogram_path:
            new_path = await asyncio.to_thread(rename_file_based_on_content, final_spectrogram_path, save_folder)
            gui_update_queue.put(('display_spectrogram', new_path))
            gui_update_queue.put(('update_filename_label', os.path.basename(new_path)))
            spectrogram_filenames.append(new_path)
            gui_update_queue.put(('update_saved_spectrograms_list',))
            last_predicted_spectrogram = new_path

            output_box1.delete(1.0, tk.END)
            bnb_result = await asyncio.to_thread(predict_and_display, new_path, output_box1)
            print(f"Debug BNB Result: {bnb_result}")

            output_box2.delete(1.0, tk.END)
            qnq_result = await asyncio.to_thread(QNQpredictor, new_path, output_box2)
            print(f"Debug QNQ Result: {qnq_result}")

            def convert_to_float(data):
                """Recursively convert all numpy float32 to float"""
                if isinstance(data, np.float32):
                    return float(data)
                elif isinstance(data, list):
                    return [convert_to_float(i) for i in data]
                elif isinstance(data, dict):
                    return {k: convert_to_float(v) for k, v in data.items()}
                return data

            bnb_result = convert_to_float(bnb_result)
            qnq_result = convert_to_float(qnq_result)
            print(f"Converted BNB Result: {bnb_result}")
            print(f"Converted QNQ Result: {qnq_result}")

            bnb_prediction = {
                "File": os.path.basename(new_path),
                "Predicted": bnb_result[0],
                "Confidence": bnb_result[1] * 100,
                "Accuracy": bnb_result[2] if len(bnb_result) > 2 else 'N/A',
                "Precision": bnb_result[3] if len(bnb_result) > 3 else 'N/A',
                "Recall": bnb_result[4] if len(bnb_result) > 4 else 'N/A',
                "F1 Score": bnb_result[5] if len(bnb_result) > 5 else 'N/A',
                "Model Retrained": bnb_result[6] if len(bnb_result) > 6 else 'N/A'
            }
            print(f"Debug BNB Prediction: {bnb_prediction}")

            qnq_prediction = {
                "File": os.path.basename(new_path),
                "Predicted": qnq_result[0],
                "Confidence": qnq_result[1] * 100,
                "Accuracy": qnq_result[2] if len(qnq_result) > 2 else 'N/A',
                "Precision": qnq_result[3] if len(qnq_result) > 3 else 'N/A',
                "Recall": qnq_result[4] if len(qnq_result) > 4 else 'N/A',
                "F1 Score": qnq_result[5] if len(qnq_result) > 5 else 'N/A',
                "Model Retrained": qnq_result[6] if len(qnq_result) > 6 else 'N/A'
            }
            print(f"Debug QNQ Prediction: {qnq_prediction}")

            discord_message = {
                "BNB Prediction": bnb_prediction,
                "QNQ Prediction": qnq_prediction
            }
            print(f"Sending to Discord: {json.dumps(discord_message, indent=2)}")
            await send_discord_message(json.dumps(discord_message, indent=2), new
                                       










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
            gui_update_queue.put(('display_spectrogram', new_path))
            gui_update_queue.put(('update_filename_label', os.path.basename(new_path)))
            spectrogram_filenames.append(new_path)
            gui_update_queue.put(('update_saved_spectrograms_list',))
            last_predicted_spectrogram = new_path

            output_box1.delete(1.0, tk.END)
            bnb_result = await asyncio.to_thread(predict_and_display, new_path, output_box1)
            print(f"Debug BNB Result: {bnb_result}")

            output_box2.delete(1.0, tk.END)
            qnq_result = await asyncio.to_thread(QNQpredictor, new_path, output_box2)
            print(f"Debug QNQ Result: {qnq_result}")

            # Define the conversion function
            def convert_to_float(data):
                """Recursively convert all numpy float32 to float."""
                if isinstance(data, np.float32):
                    return float(data)
                elif isinstance(data, np.ndarray):
                    return data.astype(float).tolist()  # Convert entire array to list of floats
                elif isinstance(data, list):
                    return [convert_to_float(i) for i in data]
                elif isinstance(data, dict):
                    return {k: convert_to_float(v) for k, v in data.items()}
                return data

            # Check if results are empty or not structured as expected
            if bnb_result is None or qnq_result is None:
                print("Error: BNB or QNQ result is None")
                return
            
            if isinstance(bnb_result, (list, tuple)) and len(bnb_result) < 2:
                print("Error: BNB result is incomplete")
                return
            
            if isinstance(qnq_result, (list, tuple)) and len(qnq_result) < 2:
                print("Error: QNQ result is incomplete")
                return

            # Convert results to standard Python types
            bnb_result = convert_to_float(bnb_result)
            qnq_result = convert_to_float(qnq_result)
            print(f"Converted BNB Result: {bnb_result}")
            print(f"Converted QNQ Result: {qnq_result}")

            # Ensure all values are standard Python types
            discord_message = {
                "BNB Prediction": {
                    "File": os.path.basename(new_path),
                    "Predicted": float(bnb_result[0]),  # Ensure this is a float
                    "Confidence": float(bnb_result[1]) * 100,  # Ensure this is a float
                    "Accuracy": float(bnb_result[2]) if len(bnb_result) > 2 else 'N/A',
                    "Precision": float(bnb_result[3]) if len(bnb_result) > 3 else 'N/A',
                    "Recall": float(bnb_result[4]) if len(bnb_result) > 4 else 'N/A',
                    "F1 Score": float(bnb_result[5]) if len(bnb_result) > 5 else 'N/A',
                    "Model Retrained": bnb_result[6] if len(bnb_result) > 6 else 'N/A'
                },
                "QNQ Prediction": {
                    "File": os.path.basename(new_path),
                    "Predicted": float(qnq_result[0]),  # Ensure this is a float
                    "Confidence": float(qnq_result[1]) * 100,  # Ensure this is a float
                    "Accuracy": float(qnq_result[2]) if len(qnq_result) > 2 else 'N/A',
                    "Precision": float(qnq_result[3]) if len(qnq_result) > 3 else 'N/A',
                    "Recall": float(qnq_result[4]) if len(qnq_result) > 4 else 'N/A',
                    "F1 Score": float(qnq_result[5]) if len(qnq_result) > 5 else 'N/A',
                    "Model Retrained": qnq_result[6] if len(qnq_result) > 6 else 'N/A'
                }
            }

            # Print the discord message for debugging
            print(f"Sending to Discord: {json.dumps(discord_message, indent=2)}")

            # Send the message to Discord
            await send_discord_message(json.dumps(discord_message, indent=2), new_path)

    except Exception as e:
        print(f"An error occurred during recording and spectrogram generation: {e}")




