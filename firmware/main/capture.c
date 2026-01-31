// capture.c
// Bradley Manzo 2026
#include "driver/gpio.h"
#include "driver/uart.h"

#include "includes/arducam.h"
#include "includes/capture.h"
#include "includes/globals.h"
#include "includes/spi.h"
#include "includes/uart.h"
#include "includes/wifi_cam.h"

#define GPIO_BYPASS_FPGA    GPIO_NUM_3
#define KERNEL_W            3

/* -------------------------------------- Packet Encoding -------------------------------------- */
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
        packet[DATA_START + packet_len + 0] = 0xFF;
        packet[DATA_START + packet_len + 1] = 0xFD;
    }
}

/* -------------------------------------- UART RX Task -------------------------------------- */

typedef struct {
    uint8_t *dst;
    size_t   cap;
    size_t   got;          // payload bytes (payload-only)
    bool     done;
    bool     error;
    uint16_t image_width;  // received width
    uint16_t image_height; // received height

    // NEW: expected minimum payload dims to reject false tails
    uint16_t expect_width;
    uint16_t expect_height;
} rx_ctx_t;

static void uart_rx_task(void *arg)
{
    rx_ctx_t *ctx = (rx_ctx_t *)arg;

    const TickType_t deadline =
        xTaskGetTickCount() + pdMS_TO_TICKS(2000);

    // State
    bool saw_a5 = false;
    int  meta_needed = 0;
    uint8_t meta[4] = {0};
    int meta_idx = 0;

    // Left-aligned payload
    size_t payload_len = 0;

    // Staging read buffer
    uint8_t rx_buf[128];

    while (true) {
        int r = uart_read_bytes(UART_NUM_1, rx_buf, (int)sizeof(rx_buf), pdMS_TO_TICKS(20));
        if (r < 0) { ctx->error = true; break; }

        if (r > 0) {
            for (int i = 0; i < r; i++) {
                uint8_t b = rx_buf[i];

                // Collecting metadata bytes after tail
                if (meta_needed > 0) {
                    meta[meta_idx++] = b;
                    meta_needed--;

                    if (meta_needed == 0) {
                        uint16_t w = ((uint16_t)meta[0] << 8) | (uint16_t)meta[1];
                        uint16_t h = ((uint16_t)meta[2] << 8) | (uint16_t)meta[3];

                        // Validate dimensions (tune bounds for your pipeline)
                        bool ok = true;
                        if (w == 0 || h == 0) ok = false;
                        if (w > QVGA_WIDTH || h > QVGA_HEIGHT) ok = false;

                        // Also validate payload size fits our buffer
                        size_t need = (((size_t)w * (size_t)h) + 7) / 8;
                        if (need > ctx->cap) ok = false;

                        // Optional: if you expect a particular output size, enforce it
                        // (keep loose to avoid false rejects during debugging)
                        if (ctx->expect_width && ctx->expect_height) {
                            // Example: allow small deviations, or require exact match:
                            // ok &= (w == ctx->expect_width && h == ctx->expect_height);
                        }

                        if (ok) {
                            ctx->image_width  = w;
                            ctx->image_height = h;
                            ctx->got = payload_len; // payload-only
                            ctx->done = true;
                            vTaskDelete(NULL);
                            return;
                        } else {
                            // False tail: push the bytes back into payload stream.
                            // We previously consumed A5 5A + 4 meta bytes but they weren't real.
                            // Emit them as payload (if room), then continue scanning.
                            const uint8_t false_seq[6] = {0xA5, 0x5A, meta[0], meta[1], meta[2], meta[3]};
                            for (int k = 0; k < 6; k++) {
                                if (payload_len < ctx->cap) ctx->dst[payload_len++] = false_seq[k];
                                else { ctx->error = true; ctx->done = true; vTaskDelete(NULL); return; }
                            }
                            // resume normal scanning
                            saw_a5 = false;
                            meta_idx = 0;
                        }
                    }
                    continue;
                }

                // If we were holding an A5
                if (saw_a5) {
                    if (b == 0x5A) {
                        // Tentatively treat as tail. Start collecting metadata.
                        meta_needed = 4;
                        meta_idx = 0;
                        saw_a5 = false;
                        continue;
                    } else {
                        // Not actually tail: emit the held A5 as payload
                        if (payload_len < ctx->cap) ctx->dst[payload_len++] = 0xA5;
                        else { ctx->error = true; break; }
                        saw_a5 = false;
                        // fall through to handle b normally
                    }
                }

                if (b == 0xA5) {
                    saw_a5 = true;
                    continue;
                }

                // Normal payload byte
                if (payload_len < ctx->cap) ctx->dst[payload_len++] = b;
                else { ctx->error = true; break; }
            }
        }

        if (ctx->error) break;

        if (xTaskGetTickCount() > deadline) {
            ctx->error = true;
            break;
        }
    }

    ctx->got = payload_len;
    ctx->done = true;
    vTaskDelete(NULL);
}

void singleCapture(void) {

    static uint8_t capture_num = 0;
    static uint8_t adaptive_th = 0;
    uint8_t downsample_factor  = 1;
    
    const uint16_t W = QVGA_WIDTH/downsample_factor;
    const uint16_t H = QVGA_HEIGHT/downsample_factor;
    const size_t tx_bytes = (((size_t)W * H + 7) / 8) + HEADER_SIZE;

    // Downsample if needed
    const uint16_t W_rx = (QVGA_WIDTH/downsample_factor)  - (KERNEL_W - 1);
    const uint16_t H_rx = (QVGA_HEIGHT/downsample_factor) - (KERNEL_W - 1);
    // Received output activation is smaller
    const size_t rx_bytes = tx_bytes; // Maximum received buffer size should be same as tx_bytes

    arducam_camlock_take();

    /* ---------------------------- Start Arducam Capture Sequence ---------------------------- */
    arducam_set_capture();
    arducam_reset_fifo();
    arducam_start_capture();
    capture_num++;
    // Poll Arducam and stop capture when done
    const TickType_t t0 = xTaskGetTickCount();
    while (!spi_get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
        // Spin without sleep for precision, or very short sleep
        // vTaskDelay(1); 
        if ((xTaskGetTickCount() - t0) > pdMS_TO_TICKS(1000)) {
            ESP_LOGE("cam", "Timeout waiting for CAP_DONE");
            spi_write_reg(ARDUCHIP_FIFO, 0x00); // Stop
            arducam_camlock_give();
            return;
        }
    }
    arducam_stop_capture();

    /* --------------------- Stream and Compress YUV422 Data from Arducam --------------------- */
    uint8_t *gray_q_tx = heap_caps_malloc(tx_bytes, MALLOC_CAP_8BIT);
    if (!gray_q_tx) {
        ESP_LOGE("cam","OOM gray_q_tx %u", (unsigned)tx_bytes);
        arducam_camlock_give();
        return;
    }
    // Reads Raw YUV422 from fifo and packs to 1bpp grayscale
    esp_err_t re = arducam_read_and_pack_stream(gray_q_tx, tx_bytes, &adaptive_th, capture_num, downsample_factor);
    
    arducam_camlock_give();

    if (re != ESP_OK) {
        ESP_LOGE("cam","read/pack failed: %s", esp_err_to_name(re));
        free(gray_q_tx);
        return;
    }

    /* ---------------- Transmit Packed 1bpp Data to FPGA for Image Processing ---------------- */
    
    // Skip fpga and publish if sampling for adaptive threshold only
    if (capture_num >= RECALIBRATE_INTERVAL) {
        free(gray_q_tx);
        capture_num = 0;
        return;
    }

    uint8_t *gray_q_rx = (uint8_t*)heap_caps_malloc(rx_bytes, MALLOC_CAP_8BIT);
    if (!gray_q_rx) {
        ESP_LOGE("main", "OOM gray_q_rx");
        free(gray_q_tx);
        return;
    }

    rx_ctx_t rx = {
        .dst = gray_q_rx,
        .cap = rx_bytes,
        .got = 0,
        .done = false,
        .error = false,
        .image_width = 0,
        .image_height = 0,
        .expect_width = W_rx,
        .expect_height = H_rx,
    };
    uart_flush_input(UART_NUM_1);
    xTaskCreatePinnedToCore(uart_rx_task, "uart_rx", 4096, &rx, 10, NULL, 0);
    
    // Necessary for proper image alignment on UART
    const size_t CHUNK_SIZE = 128; 
    size_t sent = 0;
    while (sent < tx_bytes) {
        size_t n = tx_bytes - sent;
        if (n > CHUNK_SIZE) n = CHUNK_SIZE;
        
        uart_write_bytes(UART_NUM_1, (const char*)(gray_q_tx + sent), n);
        sent += n;
        
        // Tiny delay to let FPGA drain its FIFO
        // Even 1 tick or a simple busy-wait might be enough
        esp_rom_delay_us(100); 
    }

    // Wait for RX to finish
    while (!rx.done) vTaskDelay(pdMS_TO_TICKS(10));

    if (rx.error) {
        ESP_LOGW("uart", "rx failed: got %u/%u bytes", (unsigned)rx.got, (unsigned)rx.cap);
        free(gray_q_tx);
        free(gray_q_rx);
        return;
    }
    /* ----------------------------- Package and Publish Frame ----------------------------- */
    const bool bypass = gpio_get_level(GPIO_BYPASS_FPGA);

    const uint16_t outW = bypass ? W : rx.image_width;
    const uint16_t outH = bypass ? H : rx.image_height;

    size_t payload_bytes;
    if (bypass) {
        payload_bytes = tx_bytes - HEADER_SIZE;
    } else {
        payload_bytes = (((size_t)outW * outH) + 7) / 8;

        if (payload_bytes > rx.got) payload_bytes = rx.got; // best if rx.got is payload-only
        if (payload_bytes > rx.cap) payload_bytes = rx.cap;
    }

    printf("Received height: %d, width: %d\n", rx.image_height, rx.image_width);

    size_t packet_len = DATA_START + payload_bytes + 2;
    uint8_t *packet = heap_caps_malloc(packet_len, MALLOC_CAP_8BIT);
    if (!packet) {
        ESP_LOGE("main", "packet malloc failed");
        free(gray_q_tx);
        free(gray_q_rx);
        return;
    }

    // Encode Image dimensions in header
    encapsulate(packet, 0, "header", outW, outH);
    // Encode Data length in header
    encapsulate(packet, (uint16_t)payload_bytes, "data len", outW, outH);
    // Package 1bpp data within packet body
    if (bypass) {
        // Bypass FPGA processing and send original packed data
        memcpy(packet + DATA_START, gray_q_tx + HEADER_SIZE, payload_bytes);
    } else {
        memcpy(packet + DATA_START, gray_q_rx, payload_bytes);
    }
    // Encode footer with packet end markers
    packet[DATA_START + payload_bytes + 0] = 0xFF;
    packet[DATA_START + payload_bytes + 1] = 0xFD;

    free(gray_q_tx);
    free(gray_q_rx);
    
    esp_err_t rc = publish_frame(packet, packet_len);
    if (rc == ESP_OK) ESP_LOGI("main", "published %u bytes", (unsigned)packet_len);
    else              ESP_LOGW("main", "publish failed: %d", rc);

    free(packet);
    vTaskDelay(pdMS_TO_TICKS(1));
}