#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Max JPEG your app is willing to cache/serve.
// Define elsewhere if you want a different limit.
#ifndef WIFI_CAM_MAX_JPEG
#define WIFI_CAM_MAX_JPEG  (120 * 1024)
#endif

// Bring up SoftAP + HTTP server (/, /jpg, /gray)
// SSID/pass may be NULL -> defaults used.
esp_err_t wifi_cam_init(const char *ssid, const char *pass);

// Publish a new JPEG still for /jpg (thread-safe copy).
void      wifi_cam_publish(const uint8_t *jpeg, size_t len);

// Optionally tell /gray what frame size to expect for YUV (defaults to 320x240).
void      wifi_cam_set_frame_dims(uint16_t width, uint16_t height);

// Whether the HTTP server is up.
bool      wifi_cam_started(void);

#ifdef __cplusplus
}
#endif