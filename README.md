# ESP Computer Vision

Interfacing the ESP32c3 RUST board with an Arducam 2MP camera for image processing and computer vision.

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
Dependencies limited to Espressif__esp_jpeg.

## Features So Far

- Four Color Dynamic Quantization
- RLE Compression for quantized images.
- Broadcasting live feed to html
- Motion detection by diffing successive frames
- Edge detection via 3x3 convolution with Sobel filters

## Installation

```bash
# Clone the repository
git clone git@github.com:Brmanzo/esp-computer-vision.git
cd esp-computer-vision

# Install dependencies
source <path-to-esp-idf>/export.sh
idf.py add-dependency "espressif/esp_jpeg"

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

## Usage

No UI yet, but will be added in the future.
## License

This project is licensed under the MIT License. See [esp-computer-vision/LICENSE.md](LICENSE) for details.
