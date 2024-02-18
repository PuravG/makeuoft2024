# python -m pip install -e git+https://github.com/pybluez/pybluez.git#egg=pybluez

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

sock.close()
print("Disconnected")