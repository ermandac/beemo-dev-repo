import os
import pyaudio
import numpy as np
import time
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import discord
from discord.ext import commands
import asyncio  # Import asyncio
import logging
from google.oauth2 import service_account
# Parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1              # Number of audio channels (mono)
RATE = 44100              # Sampling rate (samples per second)
CHUNK = 1024              # Number of frames per buffer
DURATION = 15             # Recording duration in seconds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("frequency")

# Google Sheets parameters
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1h387i_m0wb2RQ8zrO-gEwHPGzs0mp8qtgYjxXO7Gg00"
RANGE_NAME = "Sheet1!A:G"

# Initialize PyAudio
p = pyaudio.PyAudio()

# Function to authenticate and connect to Google Sheets
def connect_to_google_sheets():
    # Update the path to the new credentials file
    credentials_path = r'C:\Users\Carl Joseph Torres\Desktop\BeemoDos\BeemoApp\GsheetAPI\Freq-New-Client.json'
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    creds.refresh(Request())
    try:
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        return sheet
    except HttpError as err:
        logger.error(err)
        return None

# Function to add headers if they don't exist
def add_headers(sheet):
    headers = ["Date", "Time", "Average Frequency", "Highest Frequency", "Lowest Frequency", "Activity Level"]
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    values = result.get('values', [])
    if not values or values[0] != headers:
        body = {'values': [headers]}
        sheet.values().update(
            spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME,
            valueInputOption="RAW", body=body).execute()

# Function to record audio and determine activity level
async def record_and_analyze(bot, channel_id, stream):

    print("Listening...")

    # Initialize variables
    start_time = time.time()
    frequencies = []

    def callback(in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16)
        
        # Compute the FFT (Fast Fourier Transform)
        fft_data = np.fft.fft(data)
        
        # Get the frequency with the highest amplitude
        freqs = np.fft.fftfreq(len(fft_data))
        idx = np.argmax(np.abs(fft_data))
        freq = abs(freqs[idx] * RATE)
        
        frequencies.append(freq)
        
        return (in_data, pyaudio.paContinue)

    stream.start_stream()

    try:
        while time.time() - start_time < DURATION:
            await asyncio.sleep(0.1)  # Allow other tasks to run
    except KeyboardInterrupt:
        print("Stopped listening")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Calculate average, highest, and lowest frequency
    if frequencies:
        average_frequency = np.mean(frequencies)
        highest_frequency = np.max(frequencies)
        lowest_frequency = np.min(frequencies)
    else:
        average_frequency = 0
        highest_frequency = 0
        lowest_frequency = 0

    # Determine activity level
    if average_frequency < 100:
        activity_level = "No activity captured"
    elif 100 <= average_frequency <= 300:
        activity_level = "Normal activity"
    elif 300 < average_frequency <= 400:
        activity_level = "Stressed"
    else:
        activity_level = "Chaotic"

    # Get current date and time
    current_time = datetime.now()
    date_str = current_time.strftime("%Y-%m-%d")
    time_str = current_time.strftime("%H:%M:%S")

    # Save to Google Sheets
    sheet = connect_to_google_sheets()
    if sheet:
        add_headers(sheet)
        values = [[date_str, time_str, average_frequency, highest_frequency, lowest_frequency, activity_level]]
        body = {'values': values}
        result = sheet.values().append(
            spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME,
            valueInputOption="RAW", body=body).execute()
        print(f"{result.get('updates').get('updatedCells')} cells appended.")

    # Send to Discord
    message = (f"Recorded at {time_str} on {date_str}:\n"
               f"Average Frequency = {average_frequency:.2f} Hz\n"
               f"Highest Frequency = {highest_frequency:.2f} Hz\n"
               f"Lowest Frequency = {lowest_frequency:.2f} Hz\n"
               f"Activity Level = {activity_level}")
    await send_discord_message(bot, channel_id, message)

    print(message)

# Function to send a message to Discord
async def send_discord_message(bot, channel_id, message):
    channel = bot.get_channel(channel_id)
    await channel.send(message)

# Function to terminate PyAudio
def terminate_pyaudio():
    p.terminate()