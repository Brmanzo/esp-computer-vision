# ESP Computer Vision

Interfacing the ESP32c3 RUST board with an Arducam 2MP camera for image processing and computer vision.
RTL designed and tested on icebreaker V1.1a FPGA for hardware acceleration.

## Table of Contents

- [ESP Computer Vision](#esp-computer-vision)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features So Far](#features-so-far)
  - [Milestones](#milestones)
  - [Firmware Installation](#firmware-installation)
  - [Building the ESP Project](#building-the-esp-project)
  - [Hardware Installation](#hardware-installation)
    - [Synthesizing for Icebreaker Board](#synthesizing-for-icebreaker-board)
    - [Unit Testing](#unit-testing)
    - [Credits](#credits)
  - [Usage](#usage)
  - [License](#license)

## Overview

Computer vision is the process through which a machine learning model analyzes novel data and extracts meaningful information from its environment. Training these models incurs a large initial cost, then once complete can be deployed in a static configuration. When these models are mapped to Application-Specific Integrated Circuits (ASICs) they can perform inference much faster than in software due to specialized architecture and faster clock speed.
<br><br>
My goal for this project is to implement a dynamic system that captures live data and can interpret meaningful data from the real world. My current system bridges the gap between software and hardware, effectively utilizing the strengths of each domain. The ESP32c3 captures image data on its Arducam 2MP Camera, compresses each frame to one pixel per bit and transmits to the Icebreaker FPGA. Here the data image flows through the pretrained Convolutional Neural Network (CNN) resulting in a classification of the image subject. The ESP receives the result and publishes to local WiFi.

## Features So Far
- Frontend image compression
  - YUV422 Luma values interpreted as grayscale
  - Quantized from 256 values (8 bits) to as low as 2 (1 bit)
  - Downsampling from default 320x240 down to 80x60
  - Periodic auto-exposure to adjust to changing light levels
- Hardware Acceleration of Image processing via FPGA
  - Decoupled interface via UART for non-blocking transmission
  - Packet framing for proper data alignment
  - UART RTS line to ensure data integrity
  - Kernel-stationary sliding-window convolution architecture
  - Parameterized kernel size and weight widths for model flexibility
- Unit Testing of all hardware components using CocoTB
- Dynamic HTML viewing of image over ESP32 Wi-Fi

## Milestones
- **2026-01-27** — FPGA packet data framing
- **2026-01-13** — Streaming raw grayscale image
- **2026-01-03** — Unpacking hardware to support quantized bitstream
- **2025-12-18** — ESP–FPGA integration via UART loopback
- **2025-12-10** — Edge detection on FPGA using sliding-window filters
- **2025-08-22** — Quantization and run-length encoding
- **2025-08-17** — Convolution and image differencing (software)
- **2025-08-12** — Web server integration
- **2025-07-27** — JPEG image decoding from ArduCam

## Firmware Installation
Espressif toolchain version: ESP-IDF v6.1-dev-1280-gb33c9cd7ce
```bash
# Clone the repository
git clone git@github.com:Brmanzo/esp-computer-vision.git
cd esp-computer-vision

# Wiring
CS     - GPIO_NUM_7
MOSI   - GPIO_NUM_6
MISO   - GPIO_NUM_5
SCK    - GPIO_NUM_4
GND    - GND
VCC    - 3v3/5V
SDA    - GPIO_NUM_10
SCL    - GPIO_NUM_8

Button - GPIO_NUM_3
```

## Building the ESP Project
```bash
# Source the ESP-IDF Toolchain
source ~/esp/export.sh

# Open the ESP Project Directory
cd ~/esp/esp-computer-vision/firmware

# Target the ESP32c3 Board, then build and flash
idf.py set-target esp32c3
idf.py build flash monitor
```

## Hardware Installation
Yosys version: Yosys 0.57 (git sha1 3aca86049, clang++ 18.1.3 -fPIC -O3)
Netlistsvg version: 1.0.2
Rsvg-convert version: 2.58.0
CocoTB version: 1.9.1

```bash
# Within a Python virtual environment run
pip install -r requirements.txt

# Then add utilities.py to path
export PYTHONPATH="$(git rev-parse --show-toplevel)/sim/util/:$PYTHONPATH"

# Wiring
ESP32c3       icebreakerV1.1a
GPIO_NUM_21 - GPIO  4 (PMOD1A)
GPIO_NUM_20 - GPIO  2 (PMOD1A)
GPIO_NUM_1  - GPIO 47 (PMOD1A)
GND         - GND     (PMOD1B)
```
### Synthesizing for Icebreaker Board
```bash
# within repo root run
make bitstream

# then flash the resulting ice40.bin using
iceprog ice40.bin
```

### Unit Testing
```bash
# within sim/unit_testing/ open the module you'd like to test, then run
make test
```

### Credits
Arducam's [RPI Pico Cam Project](https://github.com/ArduCAM/RPI-Pico-Cam)<br>
Alex Forencich's [Verilog-Uart Interface](https://github.com/alexforencichverilog-uart)<br>
Dustin Richmond's  [CSE 225 ASIC Design Course](https://courses.engineering.ucsc.edu/courses/cse225)<br>
## Usage

No UI yet, but will be added in the future.
## License

This project is licensed under the MIT License. See [esp-computer-vision/LICENSE.md](LICENSE) for details.

