#include "freertos/FreeRTOS.h"
#include "includes/globals.h"

uint8_t *dummy_tx = NULL;
spi_device_handle_t spi_device_handle = NULL;

i2c_master_bus_handle_t bus_handle = NULL;
i2c_master_dev_handle_t camera_dev_handle = NULL;