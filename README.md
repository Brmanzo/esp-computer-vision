# ESP Computer Vision

Interfacing the ESP32c3 RUST board with an Arducam 2MP camera for image processing and computer vision.
RTL designed and tested on icebreaker V1.1a FPGA for hardware acceleration.

## Table of Contents

- [ESP Computer Vision](#esp-computer-vision)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features So Far](#features-so-far)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)

## Overview

My main goal for this project is to perform inference using the limited resources of the esp32c3 through compression and hardware acceleration.<br>
Driver software is adapted from https://github.com/ArduCAM/RPI-Pico-Cam.
Dependencies limited to Espressif__esp_jpeg and Espressif__esp32-camera.

## Features So Far

- Four Color Dynamic Quantization
- RLE Compression for quantized images.
- Broadcasting live feed to html
- Motion detection by diffing successive frames
- Edge detection via 3x3 convolution with Sobel filters

## Software Installation
Espressif toolchain version: ESP-IDF v6.1-dev-1280-gb33c9cd7ce
```bash
# Clone the repository
git clone git@github.com:Brmanzo/esp-computer-vision.git
cd esp-computer-vision

# Install dependencies
source <path-to-esp-idf>/export.sh
idf.py add-dependency "espressif/esp_jpeg espressif/esp32-camera"

# Wiring
CS   - GPIO_NUM_7
MOSI - GPIO_NUM_6
MISO - GPIO_NUM_5
SCK  - GPIO_NUM_4
GND  - GND
VCC  - 3v3/5V
SDA  - GPIO_NUM_10
SCL  - GPIO_NUM_8
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
GPIO_NUM_21 - GPIO 4 (PMOD1A)
GPIO_NUM_20 - GPIO 2 (PMOD1A)
GND         - GND    (PMOD1B)
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

## Usage

No UI yet, but will be added in the future.
## License

This project is licensed under the MIT License. See [esp-computer-vision/LICENSE.md](LICENSE) for details.

