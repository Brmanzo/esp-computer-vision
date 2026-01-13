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

// static esp_err_t ov2640_set_yuv_order_yuyv(void)
// {
//     esp_err_t err;
//     uint8_t v;

//     // Select DSP bank
//     err = i2c_write_reg(0xFF, 0x00);
//     if (err != ESP_OK) return err;

//     // 0xDA bit0 = 0
//     err = i2c_read_reg(0xDA, &v);
//     if (err != ESP_OK) return err;
//     v &= (uint8_t)~0x01;
//     err = i2c_write_reg(0xDA, v);
//     if (err != ESP_OK) return err;

//     // 0xC2 bit4 = 0
//     err = i2c_read_reg(0xC2, &v);
//     if (err != ESP_OK) return err;
//     v &= (uint8_t)~0x10;
//     err = i2c_write_reg(0xC2, v);
//     if (err != ESP_OK) return err;

//     return ESP_OK;
// }

/* Scan data received through SPI for JPEG image. */

/* --------------------------------------------- Image Processing --------------------------------------------- */
void rgb_to_gray_quantized(const uint16_t *src_rgb, uint8_t *dst_q, uint16_t w, uint16_t h)
{
    const size_t pixels = (size_t)w * h;

    int step     = 0;
    int acc      = 0;
    int byte_i   = 0;
    uint64_t avg = 0;

    // Intermediate buffer to average the grayscale values
    uint8_t *gray_pix = (uint8_t*)heap_caps_malloc(pixels, MALLOC_CAP_8BIT);
    if (!gray_pix) {
        ESP_LOGW("main","OOM gray %u bytes", (unsigned)pixels);
        vTaskDelay(pdMS_TO_TICKS(10));
        return;
    }
    
    for (size_t i = 0; i < pixels; ++i) {
        uint16_t p = src_rgb[i];
        // expand RGB565 -> 8-bit components
        uint8_t r = (p >> 11) & 0x1F; r = (r * 527 + 23) >> 6;
        uint8_t g = (p >> 5)  & 0x3F; g = (g * 259 + 33) >> 6;
        uint8_t b =  p        & 0x1F; b = (b * 527 + 23) >> 6;
        
        uint8_t gray = (uint8_t)((77 * r + 150 * g + 29 * b) >> 8); // 0..255
        avg += gray;
        gray_pix[i] = gray;
    }

    // Take the average to adaptively threshold
    avg /= pixels;
    // avg *= 1.2;  // slight bias to darkness to denoise

    for (size_t i = 0; i < pixels; ++i) {
        uint8_t q = (gray_pix[i] >= avg);  // explicit threshold
    
        acc |= (q << step);
        step++;
        if (step == 8) {
            dst_q[byte_i] = (uint8_t)acc;
            byte_i += 1;
            step = 0;
            acc = 0;
        }
    }
    if (step != 0) {
        dst_q[byte_i] = (uint8_t)acc;
    }

    free(gray_pix);
}

/* -------------------------------------- Packet Encoding -------------------------------------- */

static inline size_t pixels_to_bytes(size_t pixels) {
    return (pixels + 7) / 8; // 8 pixels per byte
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

static inline uint8_t luma_to_bit(uint8_t y) {
    return (y > 40) ? 1 : 0;  // Lowered from 127 to 40 to catch the 0x3E (62) pixels
}

esp_err_t spi_read_chunk_no_yield(uint8_t *dst, size_t n, bool keep_cs)
{
    size_t done = 0;
    while (done < n) {
        size_t to = n - done;
        if (to > RJPEG_PULL_CHUNK) to = RJPEG_PULL_CHUNK;

        spi_transaction_t t = {
            .length    = to * 8,
            .rxlength  = to * 8,
            .tx_buffer = dummy_tx, // Ensure this is zeroed or safe
            .rx_buffer = dst + done,
            .flags     = ((keep_cs || (done + to < n)) ? SPI_TRANS_CS_KEEP_ACTIVE : 0)
        };

        // BLOCKING call. No vTaskDelay here!
        esp_err_t e = spi_device_polling_transmit(spi_device_handle, &t);
        if (e != ESP_OK) return e;

        done += to;
    }
    return ESP_OK;
}

esp_err_t arducam_read_y_pack1bpp_stream(uint8_t *out, size_t out_cap, uint16_t w, uint16_t h)
{
    const size_t npix    = (size_t)w * h;
    const size_t out_len = (npix + 7) / 8;
    const size_t raw_len = npix * 2;
    if (out_cap < out_len) return ESP_ERR_NO_MEM;

    // 1. LOCK THE BUS (Critical for atomic burst)
    esp_err_t e = spi_device_acquire_bus(spi_device_handle, portMAX_DELAY);
    if (e != ESP_OK) return e;

    // 2. Start Burst Read
    uint8_t cmd = BURST_FIFO_READ;
    spi_transaction_t tc = {
        .length    = 8,
        .tx_buffer = &cmd,
        .flags     = SPI_TRANS_CS_KEEP_ACTIVE
    };
    e = spi_device_polling_transmit(spi_device_handle, &tc);
    if (e != ESP_OK) { spi_device_release_bus(spi_device_handle); return e; }

    // 3. BURN THE DUMMY BYTE (Fixes the initial phase alignment)
    uint8_t dummy_waste;
    spi_transaction_t tw = {
        .length    = 8,
        .rxlength  = 8,
        .tx_buffer = dummy_tx,
        .rx_buffer = &dummy_waste,
        .flags     = SPI_TRANS_CS_KEEP_ACTIVE
    };
    spi_device_polling_transmit(spi_device_handle, &tw);
    
    // 4. Stream Loop
    uint8_t tmp[RJPEG_PULL_CHUNK];
    size_t remaining = raw_len;
    size_t out_i = 0;
    uint8_t acc = 0;
    int bitpos = 0;
    bool keep = false; // We burned the dummy, so next byte should be Y0 (Keep)

    while (remaining) {
        size_t n = remaining > RJPEG_PULL_CHUNK ? RJPEG_PULL_CHUNK : remaining;
        
        // Pass 'true' to keep CS active until the very last chunk
        bool keep_cs = (remaining > n); 
        
        // Use the NO YIELD version
        e = spi_read_chunk_no_yield(tmp, n, keep_cs);
        // DEBUG: Print first chunk ONLY
        static bool printed_debug = false;
        if (!printed_debug) {
            printf("RAW STREAM: ");
            for(int k=0; k<16 && k<n; k++) {
                printf("%02X ", tmp[k]);
            }
            printf("\n");
            printed_debug = true;
        }
        if (e != ESP_OK) { spi_device_release_bus(spi_device_handle); return e; }

        for (size_t i = 0; i < n; i++) {
            if (keep) {
                // Using MSB first (Web Standard) - try this if LSB looks static-y
                // acc |= (luma_to_bit(tmp[i]) << (7 - bitpos)); 
                
                // Using LSB first (Your original)
                acc |= (luma_to_bit(tmp[i]) << bitpos);
                
                if (++bitpos == 8) {
                    out[out_i++] = acc;
                    acc = 0;
                    bitpos = 0;
                }
            }
            keep = !keep;
        }
        remaining -= n;
    }
    
    // Cleanup
    if (bitpos != 0 && out_i < out_len) out[out_i++] = acc;
    spi_device_release_bus(spi_device_handle);
    return ESP_OK;
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

void singleCapture(void)
{
    const uint16_t W = 320, H = 240;
    const size_t tx_bytes = ((size_t)W * H + 7) / 8;

    arducam_camlock_take();

    // 1. HARD RESET & CLEAR
    spi_write_reg(ARDUCHIP_MODE, 0x01); // Force Single Mode
    
    // Toggle the FIFO Clear bit with delays
    spi_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK); 
    esp_rom_delay_us(10); // Small hardware delay
    spi_write_reg(ARDUCHIP_FIFO, 0x00); 

    // Reset pointers
    spi_write_reg(ARDUCHIP_FIFO, FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
    spi_write_reg(ARDUCHIP_TRIG, CAP_DONE_MASK); 

    // SAFETY CHECK: Verify FIFO is empty
    // If this still fails, we just warn and continue, but the delay above should fix it.
    if (spi_read_fifo_len() > 0) {
        ESP_LOGW("cam", "FIFO stubborn! Force clearing again.");
        spi_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
        spi_write_reg(ARDUCHIP_FIFO, 0x00);
        spi_write_reg(ARDUCHIP_FIFO, FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
    }
    
    // Explicitly clear the Done Flag (using your method since it worked for you)
    spi_write_reg(ARDUCHIP_TRIG, CAP_DONE_MASK); 

    // SAFETY CHECK: Verify FIFO is empty before we start
    if (spi_read_fifo_len() > 0) {
        ESP_LOGW("cam", "FIFO not empty after reset! Force clearing again.");
        spi_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
        spi_write_reg(ARDUCHIP_FIFO, 0x00);
    }

    // ----------------------------------------------------------------------
    // 2. CAPTURE WITH TIMEOUT
    // ----------------------------------------------------------------------
    start_capture(); // Writes 0x02 to ARDUCHIP_FIFO

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

    // ----------------------------------------------------------------------
    // 3. HARD STOP
    // ----------------------------------------------------------------------
    spi_write_reg(ARDUCHIP_FIFO, 0x00); // Kill Write Enable
    spi_write_reg(ARDUCHIP_FIFO, FIFO_RDPTR_RST_MASK); // Reset Read Pointer to 0

    // ----------------------------------------------------------------------
    // 4. READ & ALLOCATE
    // ----------------------------------------------------------------------
    uint32_t len = spi_read_fifo_len();
    ESP_LOGI("cam", "Capture Done. FIFO Len: %u", (unsigned)len);

    // Allocate TX buffer
    uint8_t *gray_q_tx = heap_caps_malloc(tx_bytes, MALLOC_CAP_8BIT);
    if (!gray_q_tx) {
        ESP_LOGE("cam","OOM gray_q_tx %u", (unsigned)tx_bytes);
        arducam_camlock_give();
        return;
    }

    // Read Data
    // We ignore 'len' for the read loop to avoid buffer overflows if len is garbage
    esp_err_t re = arducam_read_y_pack1bpp_stream(gray_q_tx, tx_bytes, W, H);
    
    // Dump ASCII for debug
    debug_ascii_dump(gray_q_tx, W, H);
    
    arducam_camlock_give();

    if (re != ESP_OK) {
        ESP_LOGE("cam","read/pack failed: %s", esp_err_to_name(re));
        free(gray_q_tx);
        return;
    }

    // ----------------------------------------------------------------------
    // 5. SOFTWARE LOOPBACK (BYPASSING UART)
    // ----------------------------------------------------------------------
    // Since you are debugging the camera, we skip the FPGA/UART round-trip.
    // This allows the web server to receive the image immediately.
    
    // Packetize for HTTP publish: payload = tx_bytes, dims = W x H
    size_t packet_len = DATA_START + tx_bytes + 2;
    uint8_t *packet = heap_caps_malloc(packet_len, MALLOC_CAP_8BIT);
    if (!packet) {
        ESP_LOGE("main", "packet malloc failed");
        free(gray_q_tx);
        return;
    }

    encapsulate(packet, 0, "header", W, H);
    encapsulate(packet, (uint16_t)tx_bytes, "data len", W, H);

    // DIRECT COPY: TX -> Packet (Simulating perfect FPGA return)
    memcpy(packet + DATA_START, gray_q_tx, tx_bytes);
    
    packet[DATA_START + tx_bytes + 0] = 0xFF;
    packet[DATA_START + tx_bytes + 1] = 0xFD;
    
    free(gray_q_tx);
    // free(gray_q_rx); // Not used in bypass mode

    esp_err_t rc = publish_frame(packet, packet_len);
    if (rc == ESP_OK) ESP_LOGI("main", "published %u bytes", (unsigned)packet_len);
    else              ESP_LOGW("main", "publish failed: %d", rc);

    free(packet);
    vTaskDelay(pdMS_TO_TICKS(1));
}