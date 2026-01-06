#!/usr/bin/env python3
# Only works with images of odd widths due to YOSYS error
# Usage: python3 mag_demo.py <image_filename> <ttyUSBx>

from sys import argv
import serial
import PIL.Image as Image
import numpy as np
import os
import time
import threading

def producer() -> None:
    sent_items = 0

    while sent_items < len(byte_arrays):
        ser.write(bytes([byte_arrays[sent_items]]))
        sent_items += 1

    return

def consumer(mag_image: np.ndarray, padded_w: int, padded_h: int, pad: int) -> None:
    untrimmed = np.zeros((padded_h, padded_w), dtype=np.uint8)

    total_pixels = padded_w * padded_h
    total_bytes  = (total_pixels + 7) // 8

    byte_i = 0
    pixel_i = 0

    while byte_i < total_bytes:
        b = ser.read(1)
        if not b:
            continue

        value = b[0]

        for step in range(8):
            if pixel_i >= total_pixels:
                break
            q = (value >> step) & 0x1 # LSB first

            gray = (q << 7) & 0xFF  # Scale back to 8 bits

            r = pixel_i // padded_w
            c = pixel_i % padded_w
            untrimmed[r, c] = gray

            pixel_i += 1

        byte_i += 1
        if pixel_i % padded_w == 0:
            print(f"Processed row {pixel_i // padded_w} of {padded_h}")

    # Trim padding (pad pixels on each side)
    mag_image[:, :] = untrimmed[pad:padded_h, pad:padded_w]

# Image loading and setup
path = os.path.dirname(os.path.realpath(__file__))
if argv.__len__() < 2:
    print("Usage: python3 mag_demo.py <image_filename> <ttyUSBx>")
    exit(1)

image_name = argv[1]
if not os.path.exists(os.path.join(path, image_name)):
    print(f"Image file '{image_name}' not found.")
    exit(1)

serial_port = '/dev/ttyUSB2'  # Default port
if argv.__len__() > 2:
    serial_port = '/dev/' + argv[2]
else:
    print(f"No serial port specified. Using default: {serial_port}")
    
img_path = os.path.join(path, image_name)
img = Image.open(img_path).convert('RGB')
img_array = np.array(img)
height, width, _ = img_array.shape

pad = 1
padded_w = width + pad
padded_h = height + pad


# Serial port setup
ser = serial.Serial(port=serial_port,
                    baudrate=115200, 
                    parity=serial.PARITY_NONE, 
                    stopbits=serial.STOPBITS_ONE, 
                    bytesize=serial.EIGHTBITS,timeout=1)

# Pre-generate all byte arrays for efficiency
total_pixels = padded_w * padded_h

# Packing 8 binary inputs onto each byte
total_bytes  = (total_pixels + 7) // 8
byte_arrays  = bytearray(total_bytes)

step = 0
acc  = 0
byte_i = 0 

for row in range(padded_h):
    for col in range(padded_w):
        if row < pad or col < pad:
            q = 0
        else:
            r, g, b = img_array[row - pad, col - pad]
            gray = int(0.2989*r + 0.5870*g + 0.1140*b) & 0xFF
            q = (gray >> 7) & 0x1  # Keep only the top bit
        
        acc |= q << step # LSB first
        
        step += 1
        if step == 8:
            byte_arrays[byte_i] = acc & 0xFF
            byte_i += 1
            acc = 0
            step = 0
if step != 0:
    byte_arrays[byte_i] = acc & 0xFF

# repository image array
mag_image = np.zeros((height, width), dtype=np.uint8)
print("width,height:", width, height, "padded_w,padded_h:", padded_w, padded_h)
# Start producer and consumer threads
producer_thread = threading.Thread(target=producer, daemon=True)
consumer_thread = threading.Thread(target=consumer, args=(mag_image, padded_w, padded_h, pad), daemon=True)

ser.reset_input_buffer()
ser.reset_output_buffer()
time.sleep(0.05)

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()

filename = "magnitude_output.png"
Image.fromarray(mag_image).save(filename)
print(f"Magnitude image saved as '{filename}'")