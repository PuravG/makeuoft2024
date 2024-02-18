from drowsiness_detection.extract_face import *
import serial
import time
from bluetooth import *

esp32_bt_address = 'C8:F0:9E:01:01:12'

# Connect to the ESP32 over Bluetooth
port = 1  # The standard port for Bluetooth Serial on ESP32
sock = BluetoothSocket(RFCOMM)
sock.connect((esp32_bt_address, port))
print("Connected")

sock.send("hi" + '\n')  # Send the data with a newline character

open_cam()

twice = False

while True:
    predictions = set(open('..\drowsiness_detection\\predictions.json', 'r').read()[1:-1].split(', '))
    nose = open('..\drowsiness_detection\\nose.json', 'r').read()[1:-1].split(', ')[1]
    if 'closed' in predictions and 'yawn' in predictions:
        if twice:
            sock.send("-1" + '\n')
        else:
            twice = True
    else:
        twice = False
    sock.send(nose + '\n')

sock.close()
print("Disconnected")