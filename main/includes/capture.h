#ifndef CAPTURE_H
#define CAPTURE_H

#include <stdbool.h>
#include <stdint.h>
#include "sdkconfig.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "jpeg_decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------- Semaphore Task Isolation ---------------------------- */

/* Reserve arducam hardware. */
void arducam_camlock_take(void);

/* Release arducam hardware. */
void arducam_camlock_give(void);

/* Ensure there is a dummy transaction buffer available for FIFO burst. */
void ensure_dummy(void);
/* -------------------------------------- Image Capture -------------------------------------- */
esp_err_t arducam_read_jpeg(uint8_t *dst, size_t max_len, size_t *out_len);

/* Leverages Espressif's jpeg library to decode jpegs into pixels. */
esp_err_t jpeg_decode_from_buffer(const uint8_t *jpg_buf, size_t jpg_len,
                                  uint16_t **out_pixels, uint16_t *out_w, uint16_t *out_h,
                                  esp_jpeg_image_scale_t scale);


/* Converts returned JPEG into grayscale bitmap and reports RGB of center pixel. */
void rgb565_to_gray8(const uint16_t *src, uint8_t *dst, uint8_t *packet, uint16_t w, uint16_t h);

/* 3x3 convolution using unsigned ints to leverage ESP32c3's multipliers. */
void convolution_3x3(const uint8_t *src, uint8_t *dst,
                     uint16_t w, uint16_t h,
                     const int8_t k[3][3], int divisor, int offset);

/* Capture a single frame, process, and publish. */
void singleCapture(void);


#ifdef __cplusplus
}
#endif

#endif // CAPTURE_H