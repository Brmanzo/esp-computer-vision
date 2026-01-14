

#include <stdbool.h>
#include "esp_check.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "driver/uart.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "jpeg_decoder.h"

#include "includes/arducam.h"
#include "includes/wifi_cam.h"
#include "includes/spi.h"
#include "includes/capture.h"
#include "includes/globals.h"
#include "includes/uart.h"

#define BORDER_MARGIN 2
#define MOTION_MIN_FRAC_NUM 1
#define MOTION_MIN_FRAC_DEN 200   // >=0.5% pixels must be “subject”

/* 3x3 convolution using unsigned ints to leverage ESP32c3's multipliers. */
void convolution_3x3(const uint8_t *src, uint8_t *dst,
                     uint16_t w, uint16_t h,
                     const int8_t k[3][3], int divisor, int offset)
{
    if (!src || !dst || !w || !h || divisor == 0) return;

    // Zero / copy borders (simple choice)
    memset(dst, 0, (size_t)w * h);

    for (uint16_t y = 1; y < h - 1; y++) {
        const uint8_t *row_p = src + (y - 1) * w;
        const uint8_t *row_c = src + (y     ) * w;
        const uint8_t *row_n = src + (y + 1) * w;
        uint8_t *drow = dst + y * w;

        for (uint16_t x = 1; x < w - 1; x++) {
            int sum =
                row_p[x-1]*k[0][0] + row_p[x]*k[0][1] + row_p[x+1]*k[0][2] +
                row_c[x-1]*k[1][0] + row_c[x]*k[1][1] + row_c[x+1]*k[1][2] +
                row_n[x-1]*k[2][0] + row_n[x]*k[2][1] + row_n[x+1]*k[2][2];

            sum = sum / divisor + offset;
            if (sum < 0)   sum = 0;
            if (sum > 255) sum = 255;
            drow[x] = (uint8_t)sum;
        }
    }
}
/* For each new frame, calculates the diff between the previous, 
   drawing a bounding box around the detected movement. */
void motion_detect(uint32_t npix, uint8_t *cur_gray, uint16_t w, uint16_t h, bool have_prev, uint16_t prev_w, uint16_t prev_h, uint8_t *prev_gray, size_t prev_bytes)
{
    bool draw_box = false;
    uint16_t minx=0, miny=0, maxx=0, maxy=0;

    if (have_prev && prev_w == w && prev_h == h && prev_gray) {
        const uint8_t  tol        = DIFF_TOL;
        const uint16_t margin     = BORDER_MARGIN;
        const uint32_t min_pixels = (w*h) * MOTION_MIN_FRAC_NUM / MOTION_MIN_FRAC_DEN;

        uint16_t bx = w, by = h, ex = 0, ey = 0;  // bbox accumulator
        uint32_t subject = 0;

        for (uint32_t i = 0; i < npix; i++) {
            int diff = (int)cur_gray[i] - (int)prev_gray[i];
            if (diff < 0) diff = -diff;
            if ((uint8_t)diff > tol) {
                uint16_t x = (uint16_t)(i % w);
                uint16_t y = (uint16_t)(i / w);
                if (x <= margin || x >= (w-1-margin) || y <= margin || y >= (h-1-margin)) continue;

                // Update bounding box margins to the greatest area
                if (x < bx) bx = x;
                if (y < by) by = y;
                if (x > ex) ex = x;
                if (y > ey) ey = y;
                subject++;
            }
        }
        // If enough movement detected to consider as subject
        if (subject >= min_pixels && bx < ex && by < ey) {
            if (bx < margin) bx = margin;
            if (by < margin) by = margin;
            if (ex > w-1-margin) ex = w-1-margin;
            if (ey > h-1-margin) ey = h-1-margin;

            draw_box = true;
            minx = bx; miny = by; maxx = ex; maxy = ey;
        }
    }

    // Update prev_gray before drawing 
    if (prev_bytes < npix || prev_w != w || prev_h != h || !prev_gray) {
        free(prev_gray);
        prev_gray  = (uint8_t*)heap_caps_malloc(npix, MALLOC_CAP_8BIT);
        prev_bytes = prev_gray ? npix : 0;
        prev_w = w; prev_h = h;
    }
    if (prev_gray) {
        memcpy(prev_gray, cur_gray, npix);
        have_prev = true;
    }

    // ----------- Draw bounding box directly on cur_gray and publish -----------
    if (draw_box) {
        // horizontal edges
        for (uint16_t x = minx; x <= maxx; x++) {
            cur_gray[(size_t)miny * w + x] = 255;
            cur_gray[(size_t)maxy * w + x] = 255;
        }
        // vertical edges
        for (uint16_t y = miny; y <= maxy; y++) {
            cur_gray[(size_t)y * w + minx] = 255;
            cur_gray[(size_t)y * w + maxx] = 255;
        }
    }
}

void debug_ascii_dump(uint8_t *buffer, uint16_t w, uint16_t h) {
    // We will dump a 32x32 block from the center
    int center_x = w / 2;
    int center_y = h / 2;
    int roi_w = 32;
    int roi_h = 32;

    printf("\n--- ASCII START ---\n");
    for (int y = center_y - (roi_h/2); y < center_y + (roi_h/2); y++) {
        for (int x = center_x - (roi_w/2); x < center_x + (roi_w/2); x++) {
            // Calculate byte index. 
            // Note: Buffer is PACKED 1bpp. We must extract the bit.
            size_t pixel_index = (y * w) + x;
            size_t byte_index  = pixel_index / 8;
            int bit_pos        = pixel_index % 8; // Adjust this if you switched to MSB
            
            // Extract the bit (assuming LSB first based on your code)
            uint8_t byte = buffer[byte_index];
            uint8_t bit  = (byte >> bit_pos) & 1;

            // Print '.' for black, '#' for white
            printf("%c", bit ? '#' : '.'); 
        }
        printf("\n");
    }
    printf("--- ASCII END ---\n");
}