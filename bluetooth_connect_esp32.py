#pip install pybluez
#pip install pyserial
# python -m pip install -e git+https://github.com/pybluez/pybluez.git#egg=pybluez
# on linux : sudo apt-get install bluetooth libbluetooth-dev
# sudo python3 -m pip install pybluez
# I didn't have a specific microsoft c++ package.

import serial
import time
from bluetooth import *

# Scan for available Bluetooth devices
print("Scanning for devices...")
nearby_devices = discover_devices(lookup_names=True)
print("Found {} devices.".format(len(nearby_devices)))

# List devices
for index, device in enumerate(nearby_devices):
    addr, name = device
    print("{}: {} - {}".format(index, addr, name))

# Select your ESP32 device
selection = int(input("Select device index: "))
addr, name = nearby_devices[selection]
print("You have selected {}".format(name))

# Replace with your ESP32's Bluetooth address
# Example: '01:23:45:67:89:AB'
esp32_bt_address = 'b0:a7:32:2a:8a:46'

# Connect to the ESP32 over Bluetooth
port = 1  # The standard port for Bluetooth Serial on ESP32
sock = BluetoothSocket(RFCOMM)
sock.connect((esp32_bt_address, port))
print("Connected to {}".format(name))


# f = open("read_write_file.txt", "r")  #open the file for reading input


# Now let's send data
while True:
    # Write a string to the ESP32
    # line = f.readline().strip() # reading the file for input
    
    # data_to_send = line 
    # For Manual Input:
    data_to_send = input("Enter data to send: ")
    if data_to_send.lower() == 'exit':
        break
    sock.send(data_to_send + '\n')  # Send the data with a newline character
    time.sleep(1)  # Wait a bit, necessary for some devices

sock.close()
print("Disconnected")