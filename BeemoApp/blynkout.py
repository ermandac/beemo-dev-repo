import BlynkLib
import socket

# Your Blynk authentication token
BLYNK_AUTH = 'Lq0F3W3ovPyeXuqvtFInHhMrgUIzzs6K'

try:
    # Initialize Blynk with the correct server URL
    blynk = BlynkLib.Blynk(BLYNK_AUTH, server='sgp1.blynk.cloud', port=80)
except (socket.gaierror, ValueError) as e:
    print(f"Failed to connect to Blynk server: {e}")

# Function to send a string to a virtual pin
def send_string_to_blynk(pin, string):
    try:
        blynk.virtual_write(pin, string)
    except Exception as e:
        print(f"Failed to send data to Blynk: {e}")

# Function to run Blynk
def run_blynk():
    while True:
        try:
            blynk.run()
        except Exception as e:
            print(f"Error while running Blynk: {e}")