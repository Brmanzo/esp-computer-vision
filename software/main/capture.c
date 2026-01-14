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
    size_t expect;
    uint8_t *dst;
    volatile size_t got;
    volatile bool done;
    volatile bool error;
} rx_ctx_t;

static void uart_rx_task(void *arg)
{
    // ESP_LOGI("uart", "rx task started");
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

void singleCapture(void)
{
    static uint8_t adaptive_th = 0;
    const uint16_t W = 320, H = 240;
    const size_t tx_bytes = ((size_t)W * H + 7) / 8;

    arducam_camlock_take();

    /* ---------------------------- Start Arducam Capture Sequence ---------------------------- */
    arducam_set_capture();
    arducam_reset_fifo();
    arducam_start_capture();

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
    esp_err_t re = arducam_read_and_pack_stream(gray_q_tx, tx_bytes, W, H, &adaptive_th);
    
    arducam_camlock_give();

    if (re != ESP_OK) {
        ESP_LOGE("cam","read/pack failed: %s", esp_err_to_name(re));
        free(gray_q_tx);
        return;
    }

    /* ---------------- Transmit Packed 1bpp Data to FPGA for Image Processing ---------------- */
    
    // Skip fpga and publish if sampling for adaptive threshold only
    if (adaptive_th == 0) {
        free(gray_q_tx);
        return;
    }

    uint8_t *gray_q_rx = (uint8_t*)heap_caps_malloc(tx_bytes, MALLOC_CAP_8BIT);
    if (!gray_q_rx) {
        ESP_LOGE("main", "OOM gray_q_rx");
        free(gray_q_tx);
        return;
    }
    // Create FPGA rx task and transmit over uart
    rx_ctx_t rx = {.expect = tx_bytes,
                   .dst = gray_q_rx,
                   .got = 0, .done = false,
                   .error = false
                  };
    uart_flush_input(UART_NUM_1);
    xTaskCreatePinnedToCore(uart_rx_task, "uart_rx", 4096, &rx, 10, NULL, 0);
    
    // Send one dummy byte then all packed 1bpp data
    const uint8_t dummy[1] = {0};
    ESP_ERROR_CHECK(uart_write_all(UART_NUM_1, dummy, 1));
    ESP_ERROR_CHECK(uart_write_all(UART_NUM_1, gray_q_tx, tx_bytes));

    // Wait for RX to finish
    while (!rx.done) vTaskDelay(pdMS_TO_TICKS(10));

    if (rx.error) {
        ESP_LOGW("uart", "rx failed: got %u/%u bytes", (unsigned)rx.got, (unsigned)rx.expect);
        free(gray_q_tx);
        free(gray_q_rx);
        return;
    }
    /* ----------------------------- Package and Publish Frame ----------------------------- */
    size_t packet_len = DATA_START + tx_bytes + 2;
    uint8_t *packet = heap_caps_malloc(packet_len, MALLOC_CAP_8BIT);
    if (!packet) {
        ESP_LOGE("main", "packet malloc failed");
        free(gray_q_tx);
        free(gray_q_rx);
        return;
    }

    // Encode Image dimensions in header
    encapsulate(packet, 0, "header", W, H);
    // Encode Data length in header
    encapsulate(packet, (uint16_t)tx_bytes, "data len", W, H);
    // Package 1bpp data within packet body
    memcpy(packet + DATA_START, gray_q_rx, tx_bytes);
    // Encode footer with packet end markers
    packet[DATA_START + tx_bytes + 0] = 0xFF;
    packet[DATA_START + tx_bytes + 1] = 0xFD;
    
    free(gray_q_tx);
    free(gray_q_rx);
    
    esp_err_t rc = publish_frame(packet, packet_len);
    if (rc == ESP_OK) ESP_LOGI("main", "published %u bytes", (unsigned)packet_len);
    else              ESP_LOGW("main", "publish failed: %d", rc);

    free(packet);
    vTaskDelay(pdMS_TO_TICKS(1));
}