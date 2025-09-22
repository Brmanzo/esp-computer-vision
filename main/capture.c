
#include <stdbool.h>
#include "esp_check.h"
#include "esp_log.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "jpeg_decoder.h"

#include "includes/arducam.h"
#include "includes/wifi_cam.h"
#include "includes/spi.h"
#include "includes/capture.h"
#include "includes/globals.h"

#define MOTION_MIN_FRAC_NUM 1
#define MOTION_MIN_FRAC_DEN 200   // >=0.5% pixels must be “subject”
#define BORDER_MARGIN       2

static uint8_t  *prev_gray  = NULL;
static size_t    prev_bytes = 0;
static uint16_t  prev_w = 0, prev_h = 0;
static bool      have_prev = false;

enum packet_contents {
    HEADER_START_H,
    HEADER_START_L,
    IMAGE_HEIGHT_H,
    IMAGE_HEIGHT_L,
    IMAGE_WIDTH_H,
    IMAGE_WIDTH_L,
    PACKET_LEN_H,
    PACKET_LEN_L,
    HEADER_END_H,
    HEADER_END_L,
    DATA_START
};

/* ---------------------------- Semaphore Task Isolation ---------------------------- */
static SemaphoreHandle_t s_cam_mutex;              // camera HW lock

/* Reserve arducam hardware. */
void arducam_camlock_take(void)
{
    if (s_cam_mutex == NULL) {
        SemaphoreHandle_t m = xSemaphoreCreateRecursiveMutex();
        if (m) s_cam_mutex = m;
    }
    if (s_cam_mutex) {
        xSemaphoreTakeRecursive(s_cam_mutex, portMAX_DELAY);
    }
}
/* Release arducam hardware. */
void arducam_camlock_give(void)
{
    if (s_cam_mutex) {
        xSemaphoreGiveRecursive(s_cam_mutex);
    }
}

/* -------------------------------------- Image Capture -------------------------------------- */

/* Ensure there is a dummy transaction buffer available for FIFO burst. */
void ensure_dummy(void) {
    if (!dummy_tx) {
        dummy_tx = heap_caps_malloc(SPI_CHUNK, MALLOC_CAP_DMA);
        assert(dummy_tx);
        memset(dummy_tx, 0, SPI_CHUNK);
    }
}

/* Scan data received through SPI for JPEG image. */
esp_err_t arducam_read_jpeg(uint8_t *dst, size_t max_len, size_t *out_len) {
    *out_len = 0;
    ensure_dummy();

    // Enter burst mode, keep CS asserted
    ESP_RETURN_ON_ERROR(spi_device_acquire_bus(spi_device_handle, portMAX_DELAY), "SPI", "acquire");
    uint8_t cmd = BURST_FIFO_READ;                       // 0x3C
    spi_transaction_t t0 = {0};
        t0.length = 8;
        t0.tx_buffer = &cmd;
        t0.flags = SPI_TRANS_CS_KEEP_ACTIVE;

    esp_err_t e = spi_device_polling_transmit(spi_device_handle, &t0);
    if (e != ESP_OK) { spi_device_release_bus(spi_device_handle); return e; }

    // SOI search in buffered chunks (don’t do 1-byte transactions)
    bool have_soi = false;
    uint8_t buf[RJPEG_PULL_CHUNK];
    size_t written = 0;
    size_t scanned = 0;
    uint8_t prev = 0x00;

    while (!have_soi) {
        // Read another chunk, keep CS low
        e = spi_read_chunk(buf, sizeof(buf), true);
        if (e != ESP_OK) {
            spi_device_release_bus(spi_device_handle);
            return e;
        }
        // Scan this chunk for FF D8 (Marker + SOI)
        size_t k = 0;
        for (; k < sizeof(buf); ++k) {
            uint8_t cur = buf[k];
            // When combination is found
            if (prev == MARKER_PREFIX && cur == SOI) {
                // Check if any data after it within the buffer
                if (written + 2 > max_len) {
                    spi_device_release_bus(spi_device_handle);
                    return ESP_ERR_NO_MEM;
                }
                dst[written++] = MARKER_PREFIX;
                dst[written++] = SOI;

                // Start copying *after* the D8 just matched
                k++;
                have_soi = true;

                // Set prev to last written byte (=SOI) so EOI detection works immediately
                prev = SOI;

                // Copy the remainder of this chunk into dst while watching for EOI
                for (; k < sizeof(buf) && written < max_len; ++k) {
                    uint8_t c = buf[k];
                    dst[written++] = c;
                    if (prev == MARKER_PREFIX && c == EOI) {  // EOI
                        spi_device_release_bus(spi_device_handle);
                        *out_len = written;
                        return ESP_OK;
                    }
                    prev = c;
                }
                break;
            }
            prev = cur;
        }

        scanned += sizeof(buf);
        // If we’ve scanned way more than we can store, bail to avoid OOM
        if (!have_soi && scanned >= max_len) {
            spi_device_release_bus(spi_device_handle);
            return ESP_FAIL; // SOI never found within reasonable window
        }
    }

    // Continue reading chunk-by-chunk until EOI (FF D9) or buffer full
    while (written < max_len) {
        e = spi_read_chunk(buf, sizeof(buf), true);
        if (e != ESP_OK) { spi_device_release_bus(spi_device_handle); return e; }

        for (size_t i = 0; i < sizeof(buf) && written < max_len; ++i) {
            uint8_t c = buf[i];
            dst[written++] = c;
            if (prev == MARKER_PREFIX && c == EOI) {         // EOI
                spi_device_release_bus(spi_device_handle);
                *out_len = written;
                return ESP_OK;
            }
            prev = c;
        }
    }

    // Ran out of space without seeing EOI
    spi_device_release_bus(spi_device_handle);
    return ESP_FAIL;
}

/* Leverages Espressif's jpeg library to decode jpegs into pixels. */
esp_err_t jpeg_decode_from_buffer(const uint8_t *jpg_buf, size_t jpg_len,
                                  uint16_t **out_pixels, uint16_t *out_w, uint16_t *out_h,
                                  esp_jpeg_image_scale_t scale)
{
    *out_pixels = NULL; *out_w = *out_h = 0;

    // Config to prove jpeg for dimensions using Espressif's jpeg info helper
    esp_jpeg_image_cfg_t pcfg = {0};
        pcfg.indata      = (uint8_t*)jpg_buf;
        pcfg.indata_size = jpg_len;
        pcfg.out_format  = JPEG_IMAGE_FORMAT_RGB565;
        pcfg.out_scale   = scale;

    esp_jpeg_image_output_t info = {0};
    esp_jpeg_get_image_info(&pcfg, &info);

    uint16_t src_w = info.width;
    uint16_t src_h = info.height;
    size_t   need  = info.output_len;
    if (need == 0) {
        uint16_t w = src_w >> scale; if (!w) w = 1;
        uint16_t h = src_h >> scale; if (!h) h = 1;
        need = (size_t)w * h * 2;
    }

    // Try to allocate; if OOM, retry at a smaller scale (bigger divisor)
    esp_jpeg_image_scale_t try_scale = scale;
    uint16_t *pixels = NULL;

    for (;;) {
        pixels = (uint16_t*)heap_caps_malloc(need, MALLOC_CAP_8BIT);
        if (pixels) break;

        if (try_scale >= JPEG_IMAGE_SCALE_3) {
            return ESP_ERR_NO_MEM;
        }
        try_scale = (esp_jpeg_image_scale_t)(try_scale + 1);

        // Re-probe with new scale to get new exact output_len
        pcfg.out_scale = try_scale;
        if (esp_jpeg_get_image_info(&pcfg, &info) != ESP_OK || info.output_len == 0)
            return ESP_ERR_NO_MEM;
        need = info.output_len;
    }

    // Decode config to pass to Espressif's jpeg decoder helper
    // Row-Major Order
    esp_jpeg_image_cfg_t dcfg = {0};
        dcfg.indata      = (uint8_t*)jpg_buf;
        dcfg.indata_size = jpg_len;
        dcfg.outbuf      = (uint8_t*)pixels;
        dcfg.outbuf_size = need;
        dcfg.out_format  = JPEG_IMAGE_FORMAT_RGB565;
        dcfg.out_scale   = try_scale;
        dcfg.flags.swap_color_bytes = 0;

    esp_jpeg_image_output_t out = {0};
    esp_err_t derr = esp_jpeg_decode(&dcfg, &out);
    if (derr != ESP_OK) { free(pixels); return derr; }

    *out_pixels = pixels;
    // decoder reports *scaled* dims here
    *out_w = out.width;
    *out_h = out.height;
    return ESP_OK;
}

static inline size_t pixels_to_bytes(size_t pixels) {
    return (pixels + 3) / 4; // 4 pixels per byte
}

void rgb_to_gray(const uint16_t *src_rgb, uint8_t *dst_gray, uint16_t w, uint16_t h)
{
    const size_t pixels = (size_t)w * h;

    for (size_t i = 0; i < pixels; ++i) {
        uint16_t p = src_rgb[i];
        // expand RGB565 -> 8-bit components
        uint8_t r = (p >> 11) & 0x1F; r = (r * 527 + 23) >> 6;
        uint8_t g = (p >> 5)  & 0x3F; g = (g * 259 + 33) >> 6;
        uint8_t b =  p        & 0x1F; b = (b * 527 + 23) >> 6;

        uint8_t y = (uint8_t)((77 * r + 150 * g + 29 * b) >> 8);
        // quantize to 4 levels by selecting 2 MSBs
        uint8_t q = (y >> 6) & 0x03; // 0..3

        dst_gray[i] = (uint8_t)(q * 85); // optional gray8 store
    }
}

void encapsulate(uint8_t *packet, uint16_t packet_len, char *mode, uint16_t w, uint16_t h) {
    if (strcmp(mode, "header") == 0) {
        packet[HEADER_START_H] = 0xFF;
        packet[HEADER_START_L] = 0x01;
        packet[IMAGE_HEIGHT_H] = (uint8_t)(h >> 8);
        packet[IMAGE_HEIGHT_L] = (uint8_t)(h & 0xFF);
        packet[IMAGE_WIDTH_H]  = (uint8_t)(w >> 8);
        packet[IMAGE_WIDTH_L]  = (uint8_t)(w & 0xFF);
        packet[HEADER_END_H]    = 0xFF;
        packet[HEADER_END_L]    = 0x02; 
    } else if (strcmp(mode, "data len") == 0) {
        packet[PACKET_LEN_H] = (packet_len) >> 8;
        packet[PACKET_LEN_L] = (packet_len) & 0xFF;
    }
    else if (strcmp(mode, "footer") == 0) {
        packet[packet_len + 1] = 0xFF;
        packet[packet_len + 2] = 0xFD; 
    }
 }

/* Converts returned JPEG into grayscale bitmap and reports RGB of center pixel. */
size_t quantize(const uint8_t *src_gray, uint8_t *packet,
                             size_t packet_cap, uint16_t w, uint16_t h)
{
    const size_t pixels = (size_t)w * h;
    const size_t pixel_bytes = pixels_to_bytes(pixels);
    const size_t header_len = DATA_START;
    const size_t footer_len = 2; // EOI 0xFF, 0xFD
    const size_t needed = header_len + pixel_bytes + footer_len;

    if (packet_cap < needed) {
        ESP_LOGE("packing rgb565", "packet buffer too small: cap=%u need=%u", (unsigned)packet_cap, (unsigned)needed);
        return 0;
    }

    // zero payload area (important before ORing)
    memset(packet + header_len, 0, pixel_bytes);

    for (size_t i = 0; i < pixels; ++i) {
        // quantize to 4 levels by selecting 2 MSBs
        uint8_t q = (src_gray[i] >> 6) & 0x03; // 0..3

        const size_t byte_index = header_len + (i / 4);    // which payload byte
        const unsigned pos_within_byte = i % 4;           // 0..3
        const unsigned shift = (3 - pos_within_byte) * 2; // MSB-first: pixel0 -> bits 7..6

        packet[byte_index] |= (uint8_t)(q << shift);
    }
    return header_len + pixel_bytes + footer_len;
}

uint16_t compress_rle(uint8_t *packet, uint8_t *RLE, uint16_t w, uint16_t h) {
    uint8_t curr_color = (packet[DATA_START] >> 6) & 0x03;
    uint8_t count = 1; // first pixel
    uint16_t current_byte = DATA_START;

    for (size_t byte = DATA_START; byte < pixels_to_bytes((size_t)(w * h)); byte++) {
        for (uint8_t crumb = 0; crumb < 4; crumb++) {
            uint8_t val = (packet[byte] >> (6 - 2 * crumb)) & 0x03;
            if (val == curr_color && count < 63) {
                count++;
            } else {
                // emit run
                RLE[current_byte++] = (count << 2) | (curr_color & 0x03);
                // start new run
                curr_color = val;
                count = 1;
            }
        }
    }
    // emit last run
    if (count > 0) {
        RLE[current_byte++] = (count << 2) | (curr_color & 0x03);
    }
    return current_byte;
}

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
void motion_detect(uint32_t npix, uint8_t *cur_gray, uint16_t w, uint16_t h)
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

void singleCapture(void)
{
    // Capture into FIFO
    arducam_camlock_take();
    spi_write_reg(ARDUCHIP_MODE, CAM2LCD_MODE);  // JPEG path
    spi_reset_fifo();
    start_capture();

    // wait for capture to be done with a safety timeout
    const TickType_t t0 = xTaskGetTickCount();
    while (!spi_get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
        vTaskDelay(pdMS_TO_TICKS(1));
        if ((xTaskGetTickCount() - t0) > pdMS_TO_TICKS(250)) {
            ESP_LOGW("cam","CAP_DONE timeout");
            spi_reset_fifo();
            arducam_camlock_give();
            vTaskDelay(1);
            return;
        }
    }

    // pick a safe upper bound from the hint
    uint32_t hint = spi_read_fifo_len();
    ESP_LOGW("cam","FIFO length hint: %u", (unsigned)hint);
    if (hint == 0 || hint > WIFI_CAM_MAX_JPEG) hint = WIFI_CAM_MAX_JPEG;
    size_t cap = hint + 32; if (cap > WIFI_CAM_MAX_JPEG) cap = WIFI_CAM_MAX_JPEG;

    // Allocate buffer for JPEG from hint with extra space, throwing if OOM
    uint8_t *jpg = (uint8_t*)heap_caps_malloc(cap, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    if (!jpg) {
        ESP_LOGE("cam","OOM %u", (unsigned)cap);
        spi_reset_fifo();
        arducam_camlock_give();
        return;
    }

    size_t actual = 0;
    esp_err_t e = arducam_read_jpeg(jpg, cap, &actual);
    arducam_camlock_give();     // hardware no longer needed

    ESP_LOGI("cam","read_jpeg: %s, len=%u", esp_err_to_name(e), (unsigned)actual);
    
    // If read JPEG is valid (starts with SOI, ends with EOI)
    if (!(e == ESP_OK && actual >= 4 &&
          jpg[0] == 0xFF && jpg[1] == 0xD8 &&
          jpg[actual-2] == 0xFF && jpg[actual-1] == 0xD9)) {
        if (actual >= 2) {
            ESP_LOGW("cam","EOI missing. len=%u head=%02X %02X tail=%02X %02X",
                     (unsigned)actual, jpg[0], jpg[1], jpg[actual-2], jpg[actual-1]);
        } else {
            ESP_LOGW("cam","EOI missing. len=%u (too short)", (unsigned)actual);
        }
        free(jpg);
        vTaskDelay(pdMS_TO_TICKS(10));
        return;
    }

    // Decode JPEG from raw data into matrix of RGB565 pixels
    uint16_t *pix = NULL; uint16_t w=0, h=0;

    esp_err_t d = jpeg_decode_from_buffer(jpg, actual, &pix, &w, &h, JPEG_IMAGE_SCALE_0);
    free(jpg);   // jpg buffer not needed after decode

    if (d != ESP_OK || !pix) {
        ESP_LOGW("main","decode failed: %s", esp_err_to_name(d));
        vTaskDelay(pdMS_TO_TICKS(10));
        return;
    }
    ESP_LOGI("main","decoded %ux%u", w, h);

    // Convert to GRAY 8-bit 
    const size_t npix = (size_t)w * h;
    uint8_t *cur_gray = (uint8_t*)heap_caps_malloc(npix, MALLOC_CAP_8BIT);
    if (!cur_gray) {
        ESP_LOGW("main","OOM gray %u bytes", (unsigned)npix);
        free(pix);
        vTaskDelay(pdMS_TO_TICKS(10));
        return;
    }
    size_t pixels = (size_t)w * h;
    rgb_to_gray(pix, cur_gray, w, h);

    // Create Payload with height and width in header
    size_t payload_bytes = pixels_to_bytes(pixels);
    size_t packet_cap = DATA_START + payload_bytes + 2; // header + payload + footer
    uint8_t *packet = (uint8_t*)heap_caps_malloc((h*w) + DATA_START, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    if (!packet) {
        ESP_LOGE("main", "packet malloc failed (len=%u)", (unsigned)packet_cap);
        free(pix);
        free(cur_gray);
        return; // or handle error
    }
    encapsulate(packet, 0, "header", w, h);
    // RLE buffer (worst case: no compression)
    uint8_t *RLE = (uint8_t*)heap_caps_malloc(h*w + DATA_START, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    if (RLE) {
        encapsulate(RLE, 0, "header", w, h);
    }

    // Quantize to 4 levels and pack into packet
    size_t packet_len = quantize(cur_gray, packet, packet_cap, w, h);
    
    uint16_t end_of_data = compress_rle(packet, RLE, w, h);
    
    // Now write packet length in header
    encapsulate(RLE, end_of_data, "data len", w, h);

    // Write footer
    encapsulate(RLE, end_of_data, "footer", w, h);

    free(pix);
    // Processing decoded JPEG
    // motion_detect(npix, cur_gray, w, h);
    esp_err_t rc = publish_frame(RLE, end_of_data + DATA_START + 2);
    if (rc == ESP_OK) {
        ESP_LOGI("main", "published %u bytes", (unsigned)(end_of_data + DATA_START + 2));
    } else {
        ESP_LOGW("main", "publish failed: %d", rc);
    }

    free(cur_gray);
    free(packet);
    free(RLE);

    // Small yield for Wi-Fi/httpd 
    vTaskDelay(pdMS_TO_TICKS(10));
}
