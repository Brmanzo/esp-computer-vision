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
    uint8_t *dst;     // destination buffer
    size_t   cap;     // total buffer capacity
    size_t   got;     // bytes received so far
    bool     done;
    bool     error;
} rx_ctx_t;

static void uart_rx_task(void *arg)
{
    rx_ctx_t *ctx = (rx_ctx_t *)arg;

    const TickType_t deadline =
        xTaskGetTickCount() + pdMS_TO_TICKS(300);

    bool saw_0d = false;

    while (true) {
        int r = uart_read_bytes(
            UART_NUM_1,
            ctx->dst + ctx->got,
            ctx->cap - ctx->got,     // capacity, not expected
            pdMS_TO_TICKS(20)
        );
        if (r < 0) {
            ctx->error = true;
            break;
        }
        if (r > 0) {
            for (int i = 0; i < r; i++) {
                uint8_t b = ctx->dst[ctx->got + i];
                if (saw_0d) {
                    if (b == 0x0A) {
                        ctx->got += i + 1;   // include tail
                        ctx->done = true;
                        vTaskDelete(NULL);
                        return;
                    }
                    saw_0d = false;
                }
                if (b == 0x0D) {
                    saw_0d = true;
                }
            }
            ctx->got += (size_t)r;
        }

        if (xTaskGetTickCount() > deadline) {
            ctx->error = true;
            break;
        }

        if (ctx->got >= ctx->cap) {
            ctx->error = true; // overflow protection
            break;
        }
    }
    ctx->done = true;
    vTaskDelete(NULL);
}

void singleCapture(void)
{
    static uint8_t capture_num = 0;
    static uint8_t adaptive_th = 0;
    uint8_t downsample_factor  = 1;
    
    const uint16_t W = QVGA_WIDTH/downsample_factor;
    const uint16_t H = QVGA_HEIGHT/downsample_factor;
    const size_t tx_bytes = ((size_t)W * H + 7) / 8;

    // Downsample if needed
    const uint16_t W_rx = (QVGA_WIDTH/downsample_factor)  - (KERNEL_W - 1);
    const uint16_t H_rx = (QVGA_HEIGHT/downsample_factor) - (KERNEL_W - 1);
    // Received output activation is smaller
    const size_t rx_bytes = ((size_t)W_rx * H_rx + 7) / 8; // For FPGA processed data

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
    // Create FPGA rx task and transmit over uart
    rx_ctx_t rx = {
        .dst = gray_q_rx,
        .cap = rx_bytes,
        .got = 0, .done = false,
        .error = false
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

    const size_t payload_bytes = bypass ? tx_bytes : rx_bytes;
    const uint16_t outW = bypass ? W    : W_rx;
    const uint16_t outH = bypass ? H    : H_rx;

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
        memcpy(packet + DATA_START, gray_q_tx, tx_bytes);
    } else {
        memcpy(packet + DATA_START, gray_q_rx, rx_bytes);
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