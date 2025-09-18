
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "sdkconfig.h"

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/spi_master.h"
#include "driver/uart.h"

#include "esp_check.h"
#include "esp_log.h"
#include "esp_rom_sys.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "jpeg_decoder.h" // dependency

#include "arducam.h"
#include "ov2640.h"
#include "wifi_cam.h"

/* ---------------------------- Semaphore Task Isolation ---------------------------- */
static SemaphoreHandle_t s_cam_mutex = NULL;
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

/* -------------------------------------- SPI -------------------------------------- */
/* Initialize SPI for receiving image data from the ArduCAM-M-2MP device. */
spi_device_handle_t spi_device_handle;
static uint8_t *dummy_tx;

void spi_master_init(void) {
    spi_bus_config_t bus_config    = {};
        bus_config.mosi_io_num     = PIN_MOSI;
        bus_config.miso_io_num     = PIN_MISO;
        bus_config.sclk_io_num     = PIN_SCK;
        bus_config.quadwp_io_num   = -1;
        bus_config.quadhd_io_num   = -1;
        bus_config.max_transfer_sz = SPI_CHUNK;

    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &bus_config, SPI_DMA_CH_AUTO));

    spi_device_interface_config_t dev_config = {};
        dev_config.command_bits   = 0;
        dev_config.address_bits   = 0;
        dev_config.dummy_bits     = 0;
        dev_config.clock_speed_hz = SPI_FREQ;
        dev_config.mode           = 0;
        dev_config.spics_io_num   = PIN_CS;
        dev_config.queue_size     = 7;
        dev_config.flags          = 0;
    ESP_ERROR_CHECK(spi_bus_add_device(SPI2_HOST, &dev_config, &spi_device_handle));
}

/* Write to device register using SPI. */
void spi_write_reg(uint8_t address, uint8_t value) {
    // The data to be sent: register address with write bit, followed by the value
    uint8_t tx_buffer[2] = {address | WRITE_BIT, value};
    
    spi_transaction_t trans = {
        .length = 2 * 8, // Length is in bits
        .tx_buffer = tx_buffer,
    };
    
    // Use polling_transmit for a simple, blocking write operation
    esp_err_t err = spi_device_polling_transmit(spi_device_handle, &trans);
    ESP_ERROR_CHECK(err); // Or handle the error as needed
}

/* Read from device register using SPI. */
uint8_t spi_read_reg(uint8_t address) {
    // The data to be sent: register address with read bit, followed by a dummy byte
    uint8_t tx[2] = { (uint8_t)(address & 0x7F), 0x00 };
    uint8_t rx[2] = {0};
    spi_transaction_t t = {0};
        t.length    = 16;
        t.rxlength  = 16;
        t.tx_buffer = tx;
        t.rx_buffer = rx;
    // Use polling_transmit for a simple, blocking read operation
    esp_err_t err = spi_device_polling_transmit(spi_device_handle, &t);
    ESP_ERROR_CHECK(err);
    return rx[1];
}

/* Set bit at address using SPI. */
void spi_set_bit(unsigned char addr, unsigned char bit)
{
	unsigned char tmp;
	tmp = spi_read_reg(addr);
	spi_write_reg(addr, tmp | bit);
}
/* Clear bit at address using SPI. */
void spi_clear_bit(unsigned char addr, unsigned char bit)
{
	unsigned char tmp;
	tmp = spi_read_reg(addr);
	spi_write_reg(addr, tmp & (~bit));
}

/* Get bit at address using SPI. */
unsigned char spi_get_bit(unsigned char addr, unsigned char bit)
{
  unsigned char tmp;
  tmp = spi_read_reg(addr);
  tmp = tmp & bit;
  return tmp;
}

/* Start the image capture. */
void start_capture(void) {
	spi_write_reg(ARDUCHIP_FIFO, FIFO_START_MASK);
}

/* Ensure there is a dummy transaction buffer available for FIFO burst. */
static inline void ensure_dummy(void) {
    if (!dummy_tx) {
        dummy_tx = heap_caps_malloc(SPI_CHUNK, MALLOC_CAP_DMA);
        assert(dummy_tx);
        memset(dummy_tx, 0, SPI_CHUNK);
    }
}

/* Read the length of data in the SPI FIFO buffer. */
unsigned int spi_read_fifo_len()
{
    unsigned int len1,len2,len3,len=0;
    len1 = spi_read_reg(FIFO_SIZE1);
    len2 = spi_read_reg(FIFO_SIZE2);
    len3 = spi_read_reg(FIFO_SIZE3) & 0x7f;
    len = ((len3 << 16) | (len2 << 8) | len1) & 0x07fffff;
	return len;	
}

/* Reset the SPI FIFO buffer. */
static inline void spi_reset_fifo(void) {
    // Clear + reset read/write pointers
    spi_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK | FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
}

/* Reads exactly n bytes from SPI FIFO*/
static esp_err_t spi_read_chunk(uint8_t *dst, size_t n, bool keep_cs)
{
    // Pull exactly n bytes, yielding every RJPEG_YIELD_EVERY chunks
    size_t done = 0;
    int chunks  = 0;
    while (done < n) {
        size_t to = n - done;
        if (to > RJPEG_PULL_CHUNK) to = RJPEG_PULL_CHUNK;

        spi_transaction_t t;
        memset(&t, 0, sizeof(t));
        t.length    = to * 8;
        t.rxlength  = to * 8;
        t.tx_buffer = dummy_tx;
        t.rx_buffer = dst + done;
        t.flags     = ((keep_cs || (done + to < n)) ? SPI_TRANS_CS_KEEP_ACTIVE : 0);

        esp_err_t e = spi_device_polling_transmit(spi_device_handle, &t);
        if (e != ESP_OK) return e;

        done += to;
        chunks++;
        // let idle/Wi-Fi run
        if ((chunks % RJPEG_YIELD_EVERY) == 0) vTaskDelay(0);
    }
    return ESP_OK;
}

/* -------------------------------------- I2C -------------------------------------- */
/* Initialize i2c for configuring the ArduCAM-M-2MP device. */
i2c_master_bus_handle_t bus_handle;
i2c_master_dev_handle_t camera_dev_handle;

void i2c_master_init(){ // peripherals/i2c/i2c_basic
    // Initialize i2C bus for the camera device
    i2c_master_bus_config_t bus_config = {};
        bus_config.i2c_port          = I2C_MASTER_NUM;
        bus_config.sda_io_num        = PIN_SDA;
        bus_config.scl_io_num        = PIN_SCL;
        bus_config.clk_source        = I2C_CLK_SRC_DEFAULT;
        bus_config.glitch_ignore_cnt = 7;
        bus_config.flags.enable_internal_pullup = true;
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_config, &bus_handle));

    i2c_device_config_t dev_config = {};
    dev_config.dev_addr_length = I2C_ADDR_BIT_LEN_7;
    dev_config.scl_speed_hz = I2C_MASTER_FREQ_HZ;

    dev_config.device_address = arducam.slave_address;
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus_handle, &dev_config, &camera_dev_handle));
}


/* Read a 8-bit register from the sensor. */
int i2c_read_reg(uint8_t regID, uint8_t* regDat) {
    esp_err_t err = i2c_master_transmit_receive(
        camera_dev_handle, &regID, 1, regDat, 1, pdMS_TO_TICKS(I2C_TIMEOUT_MS)
    );
    if (err != ESP_OK) {
        ESP_LOGE("SCCB", "Read reg 0x%02X failed: %s", regID, esp_err_to_name(err));
    }
    return err;
}

/* Write an 8-bit value to an 8-bit register on the sensor. */
int i2c_write_reg(uint8_t regID, uint8_t regDat) {
    uint8_t buf[2] = {regID, regDat};
    esp_err_t err = i2c_master_transmit(camera_dev_handle, buf, 2, pdMS_TO_TICKS(I2C_TIMEOUT_MS));
    if (err != ESP_OK) {
        ESP_LOGE("SCCB", "Write reg 0x%02X=0x%02X failed: %s", regID, regDat, esp_err_to_name(err));
    }
    return err;
}

/* Write multiple 8-bit values to 8-bit registers on the sensor. */
int i2c_write_regs(const struct sensor_reg reglist[]) {
    int err = 0;
    const struct sensor_reg *next = reglist;

    // Check the next entry before processing it.
    while (next->reg != MARKER_PREFIX || next->val != MARKER_PREFIX) {
        // Pass the register and value directly to your write function.
        err = i2c_write_reg(next->reg, next->val);
        
        // If there's an error, stop and return it.
        if (err != 0) {
            return err;
        }
        // Move to the next entry in the list.
        next++;
    }
    return 0; // Return 0 for success
}

/* -------------------------------------- UART -------------------------------------- */
/* Initialize UART for streaming images to interfaces. */
volatile uint8_t cameraCommand = 0;
static QueueHandle_t uart_queue_handle;

void uart_init(void) {
    uart_config_t uart_config  = {};
        uart_config.baud_rate  = BAUD_RATE;
        uart_config.data_bits  = UART_DATA_8_BITS;
        uart_config.parity     = UART_PARITY_DISABLE;
        uart_config.stop_bits  = UART_STOP_BITS_1;
        uart_config.flow_ctrl  = UART_HW_FLOWCTRL_DISABLE;
        uart_config.source_clk = UART_SCLK_DEFAULT;

    // Install driver and ask it to create an event queue of 20 items
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, RX_BUF_SIZE, 0, QUEUE_DEPTH, &uart_queue_handle, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM, UART_TX_PIN, UART_RX_PIN,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

/* Task to handle UART events, such as receiving data. */
void uart_event_task(void *pvParameters) { // uart/events/example
    uart_event_t event;
    uint8_t* dtmp = (uint8_t*) malloc(RD_BUF_SIZE);
    
    for(;;) {
        // Wait for the next event in the queue
        if(xQueueReceive(uart_queue_handle, (void * )&event, (TickType_t)portMAX_DELAY)) {
            bzero(dtmp, RD_BUF_SIZE);
            switch(event.type) {
                // Event when UART data is received
                case UART_DATA:
                ESP_LOGI("UART_HANDLER", "[UART DATA]: %d", event.size);
                    // Read the received data from the UART buffer
                    int len = uart_read_bytes(UART_NUM, dtmp, event.size, portMAX_DELAY);
                    
                    // Echo the data back to the sender
                    uart_write_bytes(UART_NUM, (const char*) dtmp, len);

                    if (len > 0) {
                        cameraCommand = dtmp[0];
                        ESP_LOGI("UART_HANDLER", "Received new command: %c", cameraCommand);
                    }
                    break;
                
                // Other event types can be handled here
                default:
                    // Log other event types
                    ESP_LOGI("UART_HANDLER", "uart event type: %d", event.type);
                    break;
            }
        }
    }
    free(dtmp);
    dtmp = NULL;
    vTaskDelete(NULL);
}

/* -------------------------------------- Image Capture -------------------------------------- */
esp_err_t arducam_read_jpeg(uint8_t *dst, size_t max_len, size_t *out_len)
{
    *out_len = 0;
    ensure_dummy();

    // Enter burst mode, keep CS asserted
    ESP_RETURN_ON_ERROR(spi_device_acquire_bus(spi_device_handle, portMAX_DELAY), "SPI", "acquire");
    uint8_t cmd = BURST_FIFO_READ;                       // 0x3C
    spi_transaction_t t0 = { .length = 8, .tx_buffer = &cmd, .flags = SPI_TRANS_CS_KEEP_ACTIVE };
    esp_err_t e = spi_device_polling_transmit(spi_device_handle, &t0);
    if (e != ESP_OK) { spi_device_release_bus(spi_device_handle); return e; }

    // 2) SOI search in buffered chunks (don’t do 1-byte transactions)
    uint8_t buf[RJPEG_PULL_CHUNK];
    bool have_soi = false;
    size_t written = 0;
    size_t scanned = 0;
    uint8_t prev = 0x00;

    while (!have_soi) {
        // Read another chunk, keep CS low
        e = spi_read_chunk(buf, sizeof(buf), true);
        if (e != ESP_OK) { spi_device_release_bus(spi_device_handle); return e; }

        // Scan this chunk for FF D8
        size_t k = 0;
        for (; k < sizeof(buf); ++k) {
            uint8_t cur = buf[k];
            if (prev == MARKER_PREFIX && cur == SOI) {
                // Found SOI at boundary (prev,cur)
                if (written + 2 > max_len) { spi_device_release_bus(spi_device_handle); return ESP_ERR_NO_MEM; }
                dst[written++] = MARKER_PREFIX;
                dst[written++] = SOI;

                // Start copying *after* the D8 we just matched
                k++;                                  // move past D8
                have_soi = true;

                // We’ll set prev to last written byte (=SOI) so EOI detection works immediately
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
        // Guard: if we’ve scanned way more than we can store, bail
        if (!have_soi && scanned >= max_len) {
            spi_device_release_bus(spi_device_handle);
            return ESP_FAIL; // SOI never found within reasonable window
        }
    }

    // 3) Continue reading chunk-by-chunk until EOI (FF D9) or buffer full
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

    // 4) Ran out of space without seeing EOI
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


/* Converts returned JPEG into grayscale bitmap and reports RGB of center pixel. */
static void rgb565_to_gray8_with_probe(const uint16_t *src, uint8_t *dst, uint16_t w, uint16_t h){
    const size_t n = (size_t)w*h;
    const size_t center_idx = (size_t)(h/2) * w + (w/2);
    for (size_t i=0;i<n;i++){
        uint16_t p = src[i];
        uint8_t r = (p >> 11) & 0x1F; r = (r*527 + 23) >> 6;
        uint8_t g = (p >>  5) & 0x3F; g = (g*259 + 33) >> 6;
        uint8_t b =  p        & 0x1F; b = (b*527 + 23) >> 6;
        uint8_t y = (uint8_t)((77*r + 150*g + 29*b) >> 8);
        dst[i] = y;

        if (i == center_idx) {
            ESP_LOGI("main","Center RGB=(%u,%u,%u) Gray=%u", r,g,b,y);
        }
    }
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

#define MOTION_MIN_FRAC_NUM 1
#define MOTION_MIN_FRAC_DEN 200   // >=0.5% pixels must be “hot”
#define BORDER_MARGIN        2     // ignore 2px border

extern void arducam_camlock_take(void);
extern void arducam_camlock_give(void);

// persistent “previous frame” (declare once in the file, not inside the function)
static uint8_t  *prev_gray  = NULL;
static size_t    prev_bytes = 0;
static uint16_t  prev_w = 0, prev_h = 0;
static bool      have_prev = false;

void singleCapture(void)
{
    // ----------- 1) Capture into FIFO -----------
    arducam_camlock_take();
    spi_write_reg(ARDUCHIP_MODE, CAM2LCD_MODE);  // JPEG path
    spi_reset_fifo();
    start_capture();

    // wait CAP_DONE with a safety timeout
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

    if (!(e == ESP_OK && actual >= 4 &&
          jpg[0] == 0xFF && jpg[1] == 0xD8 &&
          jpg[actual-2] == 0xFF && jpg[actual-1] == 0xD9))
    {
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

    // ----------- 2) Decode JPEG to RGB565 -----------
    uint16_t *pix = NULL; uint16_t w=0, h=0;
    esp_err_t d = jpeg_decode_from_buffer(jpg, actual, &pix, &w, &h, JPEG_IMAGE_SCALE_0);
    free(jpg);   // jpg buffer not needed after decode

    if (d != ESP_OK || !pix) {
        ESP_LOGW("main","decode failed: %s", esp_err_to_name(d));
        vTaskDelay(pdMS_TO_TICKS(10));
        return;
    }
    ESP_LOGI("main","decoded %ux%u", w, h);

    const size_t npix = (size_t)w * h;

    // ----------- 3) Convert to GRAY 8-bit -----------
    uint8_t *cur_gray = (uint8_t*)heap_caps_malloc(npix, MALLOC_CAP_8BIT);
    if (!cur_gray) {
        ESP_LOGW("main","OOM gray %u bytes", (unsigned)npix);
        free(pix);
        vTaskDelay(pdMS_TO_TICKS(10));
        return;
    }
    rgb565_to_gray8_with_probe(pix, cur_gray, w, h);
    free(pix);

    // ----------- 4) Motion detect + draw box -----------
    bool   draw_box = false;
    uint16_t minx=0, miny=0, maxx=0, maxy=0;

    if (have_prev && prev_w == w && prev_h == h && prev_gray) {
        const uint8_t  tol        = DIFF_TOL;
        const uint16_t margin     = BORDER_MARGIN;
        const uint32_t min_pixels = (w*h) * MOTION_MIN_FRAC_NUM / MOTION_MIN_FRAC_DEN;

        uint16_t bx = w, by = h, ex = 0, ey = 0;  // bbox accumulator
        uint32_t hot = 0;

        for (uint32_t i = 0; i < npix; i++) {
            int diff = (int)cur_gray[i] - (int)prev_gray[i];
            if (diff < 0) diff = -diff;
            if ((uint8_t)diff > tol) {
                uint16_t x = (uint16_t)(i % w);
                uint16_t y = (uint16_t)(i / w);
                if (x <= margin || x >= (w-1-margin) || y <= margin || y >= (h-1-margin)) continue;

                if (x < bx) bx = x;
                if (y < by) by = y;
                if (x > ex) ex = x;
                if (y > ey) ey = y;
                hot++;
            }
        }

        if (hot >= min_pixels && bx < ex && by < ey) {
            // box is plausible; clamp a hair away from borders
            if (bx < margin) bx = margin;
            if (by < margin) by = margin;
            if (ex > w-1-margin) ex = w-1-margin;
            if (ey > h-1-margin) ey = h-1-margin;

            draw_box = true;
            minx = bx; miny = by; maxx = ex; maxy = ey;
        }
    }

    // ----------- 5) Update prev_gray (RAW) BEFORE drawing -----------
    if (prev_bytes < npix || prev_w != w || prev_h != h || !prev_gray) {
        free(prev_gray);
        prev_gray  = (uint8_t*)heap_caps_malloc(npix, MALLOC_CAP_8BIT);
        prev_bytes = prev_gray ? npix : 0;
        prev_w = w; prev_h = h;
    }
    if (prev_gray) {
        memcpy(prev_gray, cur_gray, npix);   // COPY RAW — no drawings
        have_prev = true;
    }

    // ----------- 6) Draw (if any) directly on cur_gray and publish -----------
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

    wifi_cam_publish_gray8_as_bmp(cur_gray, w, h);
    free(cur_gray);

    // ----------- 7) Small yield for Wi-Fi/httpd -----------
    vTaskDelay(pdMS_TO_TICKS(10));
}


/* -------------------------------------- Device Init -------------------------------------- */
/* Initialization sequence for the ArduCAM-M-2MP device. */
static void arducam_power_up_sensor(void) {
    // Enable sensor LDO
    spi_set_bit(ARDUCHIP_GPIO, GPIO_PWREN_MASK);
    vTaskDelay(pdMS_TO_TICKS(5));

    // Ensure NOT in power-down
    spi_clear_bit(ARDUCHIP_GPIO, GPIO_PWDN_MASK);
    vTaskDelay(pdMS_TO_TICKS(5));

    // Toggle reset: 0 = reset, 1 = normal operation
    spi_clear_bit(ARDUCHIP_GPIO, GPIO_RESET_MASK);
    vTaskDelay(pdMS_TO_TICKS(5));
    spi_set_bit(ARDUCHIP_GPIO, GPIO_RESET_MASK);
    vTaskDelay(pdMS_TO_TICKS(10));  // let it come up
}

/* Prepares ESP for communication and initiates image capture. */
void esp32c3_SystemInit(void) {
    i2c_master_init();
    spi_master_init();
    uart_init();
    arducam_power_up_sensor();
}

/* Detect the SPI bus operational state. */
uint8_t spiBusDetect(void){
    spi_write_reg(0x00, 0x55);
    if(spi_read_reg(0x00) == 0x55){
        printf("SPI bus normal");
        return 0;
    }else{
        printf("SPI bus error\r\n");
        return 1;
    }      
}

/* Probe the state of the OV2640 camera sensor. */
uint8_t ov2640Probe(void) {
    uint8_t id_H = 0, id_L = 0;

    // Select the bank that contains the chip ID
    // (Most OV2640 init sequences use MARKER_PREFIX=0x01 for ID reads)
    if (i2c_write_reg(MARKER_PREFIX, 0x01) != ESP_OK) {
        printf("OV2640: failed to select ID bank\r\n");
        return 1;
    }

    if (i2c_read_reg(0x0A, &id_H) != ESP_OK ||
        i2c_read_reg(0x0B, &id_L) != ESP_OK) {
        printf("OV2640: SCCB read error\r\n");
        return 1;
    }

    // Typical OV2640 IDs: 0x26 and 0x40/0x41/0x42
    if (id_H == 0x26 && (id_L == 0x40 || id_L == 0x41 || id_L == 0x42)) {
        printf("ov2640 detected (ID: 0x%02X%02X)\r\n", id_H, id_L);
        return 0;
    } else {
        printf("Can't find ov2640 sensor (read 0x%02X 0x%02X)\r\n", id_H, id_L);
        return 1;
    }
}

/* Initialize the OV2640 camera sensor. */
void ov2640Init(){
    i2c_write_reg(MARKER_PREFIX, 0x01);
    i2c_write_reg(0x12, 0x80);
    i2c_write_regs(OV2640_JPEG_INIT);
    i2c_write_regs(OV2640_YUV422);
    i2c_write_regs(OV2640_JPEG);
    i2c_write_reg(MARKER_PREFIX, 0x01);
    i2c_write_reg(0x15, 0x00);
    i2c_write_regs(OV2640_320x240_JPEG);
}

/* Set the JPEG size for the OV2640 camera. */
void OV2640_set_JPEG_size(unsigned char size)
{
	switch(size)
	{
		case res_160x120:
			i2c_write_regs(OV2640_160x120_JPEG);
			break;
		case res_176x144:
			i2c_write_regs(OV2640_176x144_JPEG);
			break;
		case res_320x240:
			i2c_write_regs(OV2640_320x240_JPEG);
			break;
		case res_352x288:
	  	i2c_write_regs(OV2640_352x288_JPEG);
			break;
		case res_640x480:
			i2c_write_regs(OV2640_640x480_JPEG);
			break;
		case res_800x600:
			i2c_write_regs(OV2640_800x600_JPEG);
			break;
		case res_1024x768:
			i2c_write_regs(OV2640_1024x768_JPEG);
			break;
		case res_1280x1024:
			i2c_write_regs(OV2640_1280x1024_JPEG);
			break;
		case res_1600x1200:
			i2c_write_regs(OV2640_1600x1200_JPEG);
			break;
		default:
			i2c_write_regs(OV2640_320x240_JPEG);
			break;
	}
}

struct camera_operate arducam = {
    .slave_address = ARDUCAM_ADDR,
    .systemInit    = esp32c3_SystemInit, // Point to your new ESP32 init function
    .busDetect     = spiBusDetect,
    .cameraProbe   = ov2640Probe,
    .cameraInit    = ov2640Init,
    .setJpegSize   = OV2640_set_JPEG_size,
};