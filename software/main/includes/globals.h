#pragma once
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "driver/spi_master.h"
#include "driver/i2c_master.h"
#include "freertos/semphr.h"

extern uint8_t *dummy_tx;                          // DMA-capable dummy TX buffer
extern spi_device_handle_t spi_device_handle;      // SPI device (ArduCAM CPLD)

extern i2c_master_bus_handle_t bus_handle;         // I2C bus
extern i2c_master_dev_handle_t camera_dev_handle;  // I2C device (sensor)

extern SemaphoreHandle_t s_cam_mutex;             // camera HW lock