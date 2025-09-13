#pragma once
#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

// Single source of truth; not a variable
#ifndef WIFI_CAM_MAX_JPEG
#define WIFI_CAM_MAX_JPEG (200 * 1024)
#endif

esp_err_t wifi_cam_init(const char *ssid, const char *pass);
void wifi_cam_publish(const uint8_t *jpeg, size_t len);