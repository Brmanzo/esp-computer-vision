#pragma once
#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

// Single source of truth; not a variable
#ifndef WIFI_CAM_MAX_JPEG
#define WIFI_CAM_MAX_JPEG (200 * 1024)
#endif

#define DEFAULT_SSID "esp-cam"
#define DEFAULT_PASS "12345678"

/* Initialize http */
esp_err_t wifi_cam_init(const char *ssid, const char *pass);
esp_err_t publish_frame(const uint8_t *data, size_t len);
