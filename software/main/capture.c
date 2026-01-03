#include <stdbool.h>
#include "esp_check.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "driver/uart.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "freertos/portmacro.h"
#include "jpeg_decoder.h"

#include "includes/arducam.h"
#include "includes/wifi_cam.h"
#include "includes/spi.h"
#include "includes/capture.h"
#include "includes/globals.h"
#include "includes/uart.h"

#define PAD 1

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
                                  uint16_t *out_padded_w, uint16_t *out_padded_h,
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

    // Try to allocate; if OOM, retry at a smaller scale (bigger divisor)
    esp_jpeg_image_scale_t try_scale = scale;
    size_t need = info.output_len;

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
    if (derr != ESP_OK) { heap_caps_free(pixels); return derr; }

    const uint16_t w = out.width;
    const uint16_t h = out.height;

    const uint16_t padded_w = w + PAD;
    const uint16_t padded_h = h + PAD;

    uint16_t *padbuf = (uint16_t*)heap_caps_calloc((size_t)padded_w * padded_h,
                                                   sizeof(uint16_t),
                                                   MALLOC_CAP_8BIT);
    if (!padbuf) { heap_caps_free(padbuf); return ESP_ERR_NO_MEM; }
    for (uint16_t y = 0; y < h; y++) {
        size_t src = (size_t)y * w;
        size_t dst = (size_t)(y + PAD) * padded_w + PAD;
        memcpy(&padbuf[dst], &pixels[src], (size_t)w * sizeof(uint16_t));
    }

    heap_caps_free(pixels);
    *out_pixels = padbuf;
    // decoder reports *scaled* dims here
    *out_w = w;
    *out_h = h;
    *out_padded_w = padded_w;
    *out_padded_h = padded_h;
    return ESP_OK;
}


/* --------------------------------------------- Image Processing --------------------------------------------- */
void rgb_to_gray_quantized(const uint16_t *src_rgb, uint8_t *dst_q, uint16_t w, uint16_t h)
{
    const size_t pixels = (size_t)w * h;

    int step   = 0;
    int acc    = 0;
    int byte_i = 0;
    
    for (size_t i = 0; i < pixels; ++i) {
        uint16_t p = src_rgb[i];
        // expand RGB565 -> 8-bit components
        uint8_t r = (p >> 11) & 0x1F; r = (r * 527 + 23) >> 6;
        uint8_t g = (p >> 5)  & 0x3F; g = (g * 259 + 33) >> 6;
        uint8_t b =  p        & 0x1F; b = (b * 527 + 23) >> 6;
        
        uint8_t gray = (uint8_t)((77 * r + 150 * g + 29 * b) >> 8) + 128;
        // quantize to 4 levels by selecting 2 MSBs
        uint8_t q = (gray >> 6) & 0x03; // 0..3

        acc = acc | (q << (step * 2));
        step++;
        if (step == 4) {
            dst_q[byte_i] = (uint8_t)acc;
            byte_i += 1;
            step = 0;
            acc = 0;
        }
    }
    if (step != 0) {
        dst_q[byte_i] = (uint8_t)acc;
    }
}

static void trim_gray_center(const uint8_t *padded, uint8_t *trimmed,
                             uint16_t w, uint16_t h, uint16_t padded_w)
{
    for (uint16_t y = 0; y < h; y++) {
        memcpy(&trimmed[(size_t)y * w],
               &padded[((size_t)(y + PAD)) * padded_w + PAD],
               w);
    }
}
/* -------------------------------------- Packet Encoding -------------------------------------- */

static inline size_t pixels_to_bytes(size_t pixels) {
    return (pixels + 3) / 4; // 4 pixels per byte
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

uint16_t compress_rle(uint8_t *packet, uint8_t *RLE, uint16_t len) {
    uint8_t curr_color = (packet[DATA_START] >> 6) & 0x03;
    uint8_t count = 1; // first pixel
    uint16_t current_byte = DATA_START;

    for (size_t byte = DATA_START; byte < len; byte++) {
        if (curr_color == 0b00 && byte == 0b00000000 ||
            curr_color == 0b01 && byte == 0b01010101 ||
            curr_color == 0b10 && byte == 0b10101010 ||
            curr_color == 0b11 && byte == 0b11111111) {
            count += 4;
        }
        else {
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
            continue;
        }
    }
    // emit last run
    if (count > 0) {
        RLE[current_byte++] = (count << 2) | (curr_color & 0x03);
    }
    return current_byte;
}

/* -------------------------------------- UART RX Task -------------------------------------- */

typedef struct {
    size_t expect;
    uint8_t *dst;
    volatile size_t got;
    volatile bool done;
    volatile bool error;
} rx_ctx_t;

static void uart_rx_task(void *arg)
{
    ESP_LOGI("uart", "rx task started");
    rx_ctx_t *ctx = (rx_ctx_t*)arg;
    const TickType_t deadline = xTaskGetTickCount() + pdMS_TO_TICKS(4000);

    while (ctx->got < ctx->expect) {
        int r = uart_read_bytes(UART_NUM_1,
                                ctx->dst + ctx->got,
                                ctx->expect - ctx->got,
                                pdMS_TO_TICKS(50));
        if (r < 0) { ctx->error = true; break; }
        if (r > 0) ctx->got += (size_t)r;

        if (xTaskGetTickCount() > deadline) { ctx->error = true; break; }
    }
    ctx->done = true;
    vTaskDelete(NULL);
}

/* Receives jpeg from arducam, transmits to and receives processed image from icebreaker, then publishes to esp Wi-Fi */
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

    /* ------------------------------------------------ Read Camera for Data ------------------------------------------------ */
    // Allocate buffer for JPEG from hint with extra space, throwing if OOM
    uint8_t *jpg = (uint8_t*)heap_caps_malloc(cap, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    if (!jpg) {
        ESP_LOGE("cam","OOM %u", (unsigned)cap);
        spi_reset_fifo();
        arducam_camlock_give();
        return;
    }
    size_t actual = 0;
    // Read JPEG data from camera FIFO
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

    /* ---------------------------------------------- Decode JPEG from Camera Data ---------------------------------------------- */
    uint16_t *pix = NULL; uint16_t w=0, h=0, padded_w=0, padded_h=0;

    esp_err_t d = jpeg_decode_from_buffer(jpg, actual, &pix, &w, &h, &padded_w, &padded_h, JPEG_IMAGE_SCALE_0);
    free(jpg);   // jpg buffer not needed after decode

    if (d != ESP_OK || !pix) {
        ESP_LOGW("main","decode failed: %s", esp_err_to_name(d));
        vTaskDelay(pdMS_TO_TICKS(10));
        return;
    }
    ESP_LOGI("main","decoded %ux%u", w, h);
    
    /* ------------------------------------------------- Convert to Gray and Quantize ------------------------------------------------- */
    const size_t padded_pixels = (size_t)(padded_w * padded_h);
    size_t padded_bytes = pixels_to_bytes(padded_pixels);
    ESP_LOGI("dims", "w=%u h=%u padded_w=%u padded_h=%u padded_pixels=%u",
         w, h, padded_w, padded_h, (unsigned)padded_pixels);
    uint8_t *gray_q_tx = (uint8_t*)heap_caps_malloc(padded_bytes, MALLOC_CAP_8BIT);
    if (!gray_q_tx) {
        ESP_LOGW("main","OOM gray %u bytes", (unsigned)padded_pixels);
        free(pix);
        vTaskDelay(pdMS_TO_TICKS(10));
        return;
    }
    rgb_to_gray_quantized(pix, gray_q_tx, padded_w, padded_h);

    /* ----------------------------------------------- Async Transmit Receive to FPGA ----------------------------------------------- */
    uint8_t *gray_q_rx = (uint8_t*)heap_caps_malloc(padded_bytes, MALLOC_CAP_8BIT);
    if (!gray_q_rx) {
        ESP_LOGE("main", "OOM gray_q_rx");
        free(gray_q_rx);
        return;
    }
    rx_ctx_t rx = {.expect = padded_bytes, .dst = gray_q_rx, .got = 0, .done = false, .error = false};

    uart_flush_input(UART_NUM_1);
    xTaskCreatePinnedToCore(uart_rx_task, "uart_rx", 4096, &rx, 10, NULL, 0);

    const uint8_t dummy[1] = {0};
    ESP_ERROR_CHECK(uart_write_all(UART_NUM_1, dummy, 1));
    ESP_ERROR_CHECK(uart_write_all(UART_NUM_1, gray_q_tx, padded_pixels));

    // Wait for RX to finish
    while (!rx.done) vTaskDelay(pdMS_TO_TICKS(10));

    if (rx.error) {
        ESP_LOGW("uart", "rx failed: got %u/%u bytes", (unsigned)rx.got, (unsigned)rx.expect);
        // cleanup + return
    }

    free(gray_q_tx);

    /* ----------------------------------------------- Encode and Encapsulate Packket ----------------------------------------------- */
    // Create Payload with height and width in header
    size_t payload_bytes = pixels_to_bytes(gray_q_rx);
    size_t packet_cap = DATA_START + payload_bytes + 2; // header + payload + footer
    uint8_t *packet = (uint8_t*)heap_caps_malloc((gray_q_rx) + DATA_START, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    if (!packet) {
        ESP_LOGE("main", "packet malloc failed (len=%u)", (unsigned)packet_cap);
        free(pix);
        return; // or handle error
    }
    encapsulate(packet, 0, "header", padded_w, padded_h);
    // RLE buffer (worst case: no compression)
    uint8_t *RLE = (uint8_t*)heap_caps_malloc(padded_bytes + DATA_START, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    if (RLE) {
        encapsulate(RLE, 0, "header", padded_w, padded_h);
    }

    // Quantize to 4 levels and pack into packet
    // size_t packet_len = quantize(gray_trimmed_frame, packet, packet_cap, trimmed_w, trimmed_h);
    
    uint16_t end_of_data = compress_rle(packet, RLE, padded_bytes);
    
    // Now write packet length in header
    encapsulate(RLE, end_of_data, "data len", padded_w, padded_h);

    // Write footer
    encapsulate(RLE, end_of_data, "footer", padded_w, padded_h);

    free(pix);
    // Processing decoded JPEG
    /* ------------------------------------------------------ Publish Frame ------------------------------------------------------ */
    esp_err_t rc = publish_frame(RLE, end_of_data + DATA_START + 2);
    if (rc == ESP_OK) {
        ESP_LOGI("main", "published %u bytes", (unsigned)(end_of_data + DATA_START + 2));
    } else {
        ESP_LOGW("main", "publish failed: %d", rc);
    }
    free(gray_trimmed_frame);
    free(packet);
    free(RLE);

    // Small yield for Wi-Fi/httpd 
    vTaskDelay(pdMS_TO_TICKS(10));
}
