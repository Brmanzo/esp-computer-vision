// wifi_cam.h
// Bradley Manzo 2026
#pragma once
#include "esp_err.h"

// Single source of truth; not a variable
#ifndef WIFI_CAM_MAX_JPEG
#define WIFI_CAM_MAX_JPEG (200 * 1024)
#endif

#define DEFAULT_SSID "esp-cam"
#define DEFAULT_PASS "12345678"

/* Initialize wifi task for publishing camera frames */
esp_err_t wifi_cam_init(const char *ssid, const char *pass);
/* Publish a single frame over Wi-Fi */
esp_err_t publish_frame(const uint8_t *data, size_t len);
