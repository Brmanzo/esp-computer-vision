// globals.c
// Bradley Manzo 2026
#include "includes/globals.h"

uint8_t *dummy_tx = NULL;
spi_device_handle_t spi_device_handle = NULL;

i2c_master_bus_handle_t i2c_bus_handle = NULL;
i2c_master_dev_handle_t camera_dev_handle = NULL;

SemaphoreHandle_t s_cam_mutex = NULL; 