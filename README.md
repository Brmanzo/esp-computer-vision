# ESP Computer Vision

Interfacing the ESP32c3 RUST board with an Arducam 2MP camera for image processing and computer vision.
RTL designed and tested on icebreaker V1.1a FPGA for hardware acceleration.

## Table of Contents


- [ESP Computer Vision](#esp-computer-vision)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features So Far](#features-so-far)
  - [Milestones](#milestones)
  - [Bill of Materials](#bill-of-materials)
  - [Firmware Installation](#firmware-installation)
    - [Building the ESP Project](#building-the-esp-project)
  - [Hardware Installation](#hardware-installation)
    - [Synthesizing for Icebreaker Board](#synthesizing-for-icebreaker-board)
    - [Unit Testing](#unit-testing)
  - [Credits](#credits)
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
- **2026-02-05** — Convolution Layer with N Channels
- **2026-01-27** — FPGA packet data framing
- **2026-01-13** — Streaming raw grayscale image
- **2026-01-03** — Unpacking hardware to support quantized bitstream
- **2025-12-18** — ESP–FPGA integration via UART loopback
- **2025-12-10** — Edge detection on FPGA using sliding-window filters
- **2025-08-22** — Quantization and run-length encoding
- **2025-08-17** — Convolution and image differencing (software)
- **2025-08-12** — Web server integration
- **2025-07-27** — JPEG image decoding from ArduCam

## Bill of Materials
- 1 [ESP32c3 RUST Dev Board](https://www.digikey.com/en/products/detail/espressif-systems/ESP32-C3-DEVKIT-RUST-1/17883272)
- 1 [Arducam Mini 2MP Plus - OV2640 SPI Camera Module](https://www.arducam.com/arducam-2mp-spi-camera-b0067-arduino.html)
- 1 [Icebreaker FPGA V1.1a](https://1bitsquared.com/products/icebreaker?srsltid=AfmBOooW6iLzoyD-4AgGGWnSkeym8laQqy-KYlYx5T9ydV_uMVwLNnr3)
- 1 [Tactile Switch](https://www.adafruit.com/product/367?srsltid=AfmBOopXJDkwZZ3xhYzoRondXTCAUdlD5mo5Dscsqsx_aSRLQ7fGItza)
- 1 [10KΩ Resistor](https://www.mouser.com/ProductDetail/KOA-Speer/MF1-4DCT52R1002D?qs=N%252ByFMz%252BPAuUNYagYAcIWwQ%3D%3D&srsltid=AfmBOopf6QByeJrdJKKnI24gIPw9OYOuHk75J2uJObQ4Vzyd8SYryvey)
- [Wire Jumpers](https://www.mouser.com/ProductDetail/Bud-Industries/BC-32626?qs=35lE6QEawPl6Nhk5Cl2jpw%3D%3D&mgh=1&utm_id=22173219802&utm_source=google&utm_medium=cpc&utm_marketing_tactic=amermsp&gad_source=1&gad_campaignid=22282221042&gclid=CjwKCAiA1obMBhAbEiwAsUBbIheD_8AtmTq2KaObIIA5rpbOFelhm25I9fOf8Q4eUZeYwtdDEaLhrxoCDswQAvD_BwE)

## Firmware Installation
| Component | Version |
|----------|---------|
|Espressif toolchain | v6.1 (dev-1280-gb33c9cd7ce)|

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

### Building the ESP Project
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
| Component | Version |
|----------|---------|
| Yosys | 0.57 (git 3aca86049) |
| nextpnr-ice40 | 0.6-3build5 |
| Verilator | 5.020 |
| Icarus Verilog | 12.0 |
| Verible | v0.0-4051-g9fdb |
| cocotb | 1.9.1 |
| Python | 3.12.3 |
| Netlistsvg | 1.0.2 |
| librsvg (rsvg-convert) | 2.58.0 |

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

## Credits
|Author| Source|
|----------|---------|
|Arducam | [RPI Pico Cam Project](https://github.com/ArduCAM/RPI-Pico-Cam) |
|Alex Forencich | [Verilog-Uart Interface](https://github.com/alexforencichverilog-uart)|
|Dustin Richmond |  [CSE 225 ASIC Design Course](https://courses.engineering.ucsc.edu/courses/cse225)|

## License

This project is licensed under the MIT License. See [esp-computer-vision/LICENSE.md](LICENSE) for details.

