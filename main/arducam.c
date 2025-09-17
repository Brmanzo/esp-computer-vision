
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/i2c_master.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_rom_sys.h"
#include "driver/uart.h"
#include "jpeg_decoder.h"
#include "freertos/semphr.h"

#include "arducam.h"
#include "ov2640.h"
#include "wifi_cam.h"

#define IMAGE_W 320
#define IMAGE_H 240

void spi_set_bit(unsigned char addr, unsigned char bit);
void spi_clear_bit(unsigned char addr, unsigned char bit);

i2c_master_bus_handle_t bus_handle;
i2c_master_dev_handle_t camera_dev_handle;

spi_device_handle_t spi_device_handle;

volatile uint8_t cameraCommand = 0;
static QueueHandle_t uart_queue_handle;
static const char* TAG = "UART_HANDLER";
static uint8_t *dummy_tx;

static SemaphoreHandle_t s_cam_mutex = NULL;

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

void arducam_camlock_give(void)
{
    if (s_cam_mutex) {
        xSemaphoreGiveRecursive(s_cam_mutex);
    }
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

/* Pull exactly n bytes from the ArduCAM burst stream.
   If keep_cs is true, CS stays asserted after this transfer. */
esp_err_t spi_pull_bytes(uint8_t *buf, size_t n, bool keep_cs)
{
    spi_transaction_t t;
    memset(&t, 0, sizeof(t));
    t.length    = n * 8;
    t.rxlength  = n * 8;
    t.tx_buffer = dummy_tx;
    t.rx_buffer = buf;
    t.flags     = keep_cs ? SPI_TRANS_CS_KEEP_ACTIVE : 0;
    return spi_device_polling_transmit(spi_device_handle, &t);
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

/* Initialize i2c for configuring the ArduCAM-M-2MP device. */
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

/* Initialize SPI for receiving image data from the ArduCAM-M-2MP device. */
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

/* Initialize UART for streaming images to interfaces. */
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
                ESP_LOGI(TAG, "[UART DATA]: %d", event.size);
                    // Read the received data from the UART buffer
                    int len = uart_read_bytes(UART_NUM, dtmp, event.size, portMAX_DELAY);
                    
                    // Echo the data back to the sender
                    uart_write_bytes(UART_NUM, (const char*) dtmp, len);

                    if (len > 0) {
                        cameraCommand = dtmp[0];
                        ESP_LOGI(TAG, "Received new command: %c", cameraCommand);
                    }
                    break;
                
                // Other event types can be handled here
                default:
                    // Log other event types
                    ESP_LOGI(TAG, "uart event type: %d", event.type);
                    break;
            }
        }
    }
    free(dtmp);
    dtmp = NULL;
    vTaskDelete(NULL);
}

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
    while (next->reg != 0xff || next->val != 0xff) {
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

/* Read a byte from the SPI FIFO buffer. */
unsigned char read_fifo(void) {
	unsigned char data;
	data = spi_read_reg(SINGLE_FIFO_READ);
	return data;
}

/* Flush the SPI FIFO buffer. */
void flush_fifo(void) {
	spi_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
}

/* Clear the SPI FIFO flag. */
void clear_fifo_flag(void ) {
    spi_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
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
void set_fifo_burst(uint8_t *buffer, uint32_t length) {
    ensure_dummy();

    // Keep the hardware bus exclusively during the whole burst
    ESP_ERROR_CHECK(spi_device_acquire_bus(spi_device_handle, portMAX_DELAY));

    // Send BURST command, keep CS low
    uint8_t cmd = BURST_FIFO_READ;
    spi_transaction_t t0 = (spi_transaction_t){0};
    t0.length    = 8;
    t0.tx_buffer = &cmd;
    t0.flags     = SPI_TRANS_CS_KEEP_ACTIVE;
    ESP_ERROR_CHECK(spi_device_polling_transmit(spi_device_handle, &t0));

    //Read all data in 4092-byte chunks; keep CS low until last
    while (length > 0) {
        uint32_t n = (length > SPI_CHUNK) ? SPI_CHUNK : length;

        spi_transaction_t t = (spi_transaction_t){0};
        t.length    = n * 8;
        t.rxlength  = n * 8;
        t.tx_buffer = dummy_tx;
        t.rx_buffer = buffer;
        t.flags     = (n == length) ? 0 : SPI_TRANS_CS_KEEP_ACTIVE;

        ESP_ERROR_CHECK(spi_device_polling_transmit(spi_device_handle, &t));

        buffer += n;
        length -= n;
    }

    spi_device_release_bus(spi_device_handle);
}

/* Read the length of data in the SPI FIFO buffer. */
unsigned int read_fifo_length()
{
    unsigned int len1,len2,len3,len=0;
    len1 = spi_read_reg(FIFO_SIZE1);
    len2 = spi_read_reg(FIFO_SIZE2);
    len3 = spi_read_reg(FIFO_SIZE3) & 0x7f;
    len = ((len3 << 16) | (len2 << 8) | len1) & 0x07fffff;
	return len;	
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
/* Initialize the OV2640 camera sensor. */
void ov2640Init(){
    i2c_write_reg(0xff, 0x01);
    i2c_write_reg(0x12, 0x80);
    i2c_write_regs(OV2640_JPEG_INIT);
    i2c_write_regs(OV2640_YUV422);
    i2c_write_regs(OV2640_JPEG);
    i2c_write_reg(0xff, 0x01);
    i2c_write_reg(0x15, 0x00);
    i2c_write_regs(OV2640_320x240_JPEG);
}
/* Reset the SPI FIFO buffer. */
static inline void fifo_reset_all(void) {
    // Clear + reset read/write pointers
    spi_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK | FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
}

esp_err_t arducam_read_jpeg(uint8_t *dst, size_t max_len, size_t *out_len)
{
    *out_len = 0;
    ensure_dummy();

    // Enter burst mode and keep CS low
    ESP_ERROR_CHECK(spi_device_acquire_bus(spi_device_handle, portMAX_DELAY));
    uint8_t cmd = BURST_FIFO_READ;            // 0x3C
    spi_transaction_t t0;
    memset(&t0, 0, sizeof(t0));
    t0.length    = 8;
    t0.tx_buffer = &cmd;
    t0.flags     = SPI_TRANS_CS_KEEP_ACTIVE;
    ESP_ERROR_CHECK(spi_device_polling_transmit(spi_device_handle, &t0));

    // Some boards output a dummy byte after 0x3C; some don't.
    // Don't guess — scan for SOI (0xFF 0xD8).
    uint8_t prev = 0x00, cur = 0x00;
    bool have_soi = false;
    size_t written = 0;

    // 1) Synchronize to SOI
    while (!have_soi && written < max_len) {
        ESP_ERROR_CHECK(spi_pull_bytes(&cur, 1, true));
        if (prev == 0xFF && cur == 0xD8) {
            if (written + 2 > max_len) { spi_device_release_bus(spi_device_handle); return ESP_FAIL; }
            dst[written++] = 0xFF;
            dst[written++] = 0xD8;
            have_soi = true;
            break;
        }
        prev = cur;
    }

    if (!have_soi) { spi_device_release_bus(spi_device_handle); return ESP_FAIL; }

    // 2) Copy until EOI (FF D9)
    const size_t CHUNK = 1024;
    uint8_t buf[CHUNK];
    bool done = false;
    prev = 0x00;

    while (!done && written < max_len) {
        size_t need = (max_len - written) < CHUNK ? (max_len - written) : CHUNK;

        // Keep CS low between chunks; we’ll release on the last call or after loop
        ESP_ERROR_CHECK(spi_pull_bytes(buf, need, true));

        for (size_t i = 0; i < need; ++i) {
            cur = buf[i];
            dst[written++] = cur;

            if (prev == 0xFF && cur == 0xD9) {
                done = true;
                break;
            }
            prev = cur;

            if (written == max_len) break; // guard
        }
    }

    // Final small transaction to let CS go high 
    spi_device_release_bus(spi_device_handle);
    // EOI not found within max_len
    if (!done) return ESP_FAIL;

    *out_len = written;
    return ESP_OK;
}

static bool jpeg_get_dims(const uint8_t *p, size_t n, uint16_t *w, uint16_t *h) {
    if (n < 4 || p[0] != 0xFF || p[1] != 0xD8) return false; // SOI
    size_t i = 2;
    while (i + 3 < n) {
        if (p[i] != 0xFF) { i++; continue; }
        while (i < n && p[i] == 0xFF) i++;        // fill bytes
        if (i >= n) break;
        uint8_t marker = p[i++];
        if (marker == 0xD8 || marker == 0xD9) continue; // SOI/EOI
        if (i + 2 > n) break;
        uint16_t seglen = ((uint16_t)p[i] << 8) | p[i+1];
        i += 2;
        if (seglen < 2 || i + (seglen - 2) > n) break;
        if (marker == 0xC0 || marker == 0xC2) {          // SOF0/SOF2
            if (seglen < 7) break;
            uint16_t H = ((uint16_t)p[i+1] << 8) | p[i+2];
            uint16_t W = ((uint16_t)p[i+3] << 8) | p[i+4];
            if (w) *w = W;
            if (h) *h = H;
            return true;
        }
        i += (seglen - 2);
    }
    return false;
}

esp_err_t jpeg_decode_from_buffer(const uint8_t *jpg_buf, size_t jpg_len,
                                  uint16_t **out_pixels, uint16_t *out_w, uint16_t *out_h,
                                  esp_jpeg_image_scale_t scale)
{
    *out_pixels = NULL; *out_w = *out_h = 0;

    uint16_t src_w=0, src_h=0;
    if (!jpeg_get_dims(jpg_buf, jpg_len, &src_w, &src_h)) {
        ESP_LOGE("jpeg", "Failed to parse JPEG dimensions");
        return ESP_FAIL;
    }

    // Pick scaled size (decoder will report exact size back too)
    uint16_t w = src_w >> scale; if (!w) w = 1;
    uint16_t h = src_h >> scale; if (!h) h = 1;

    size_t out_bytes = (size_t)w * h * 2; // RGB565 bytes
    uint16_t *pixels = (uint16_t*)heap_caps_malloc(out_bytes, MALLOC_CAP_8BIT);
    if (!pixels) {
        ESP_LOGE("jpeg", "OOM: %u bytes", (unsigned)out_bytes);
        return ESP_ERR_NO_MEM;
    }

    esp_jpeg_image_cfg_t cfg = {0};
        cfg.indata      = (uint8_t*)jpg_buf;
        cfg.indata_size = jpg_len;
        cfg.outbuf      = (uint8_t*)pixels;
        cfg.outbuf_size = out_bytes;
        cfg.out_format  = JPEG_IMAGE_FORMAT_RGB565;     // portable
        cfg.out_scale   = scale;
        cfg.flags.swap_color_bytes = 0;
    esp_jpeg_image_output_t out = {0};
    esp_err_t err = esp_jpeg_decode(&cfg, &out);
    if (err != ESP_OK) { free(pixels); return err; }

    *out_pixels = pixels;
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

void singleCapture(void)
{
    esp_err_t e;
    size_t    actual = 0;
    uint8_t  *jpg = NULL;

    // ---- CRITICAL SECTION: camera HW only ----
    arducam_camlock_take();

    // Select JPEG path in CPLD *before* capture
    spi_write_reg(ARDUCHIP_MODE, CAM2LCD_MODE);

    fifo_reset_all();
    start_capture();

    // Short wait loop with yield (prevents WDT while holding the lock)
    TickType_t t0 = xTaskGetTickCount();
    while (!spi_get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
        vTaskDelay(pdMS_TO_TICKS(1));
        if ((xTaskGetTickCount() - t0) > pdMS_TO_TICKS(250)) {
            ESP_LOGW("cam","CAP_DONE timeout");
            fifo_reset_all();
            arducam_camlock_give();   // <-- never forget to release
            return;
        }
    }

    // Size hint only (don’t trust for JPEG), choose a safe cap
    uint32_t hint = read_fifo_length();
    if (hint == 0 || hint > WIFI_CAM_MAX_JPEG) hint = WIFI_CAM_MAX_JPEG;
    size_t max_len = hint + 32;
    if (max_len > WIFI_CAM_MAX_JPEG) max_len = WIFI_CAM_MAX_JPEG;

    jpg = heap_caps_malloc(max_len, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    if (!jpg) {
        ESP_LOGE("cam","OOM %u", (unsigned)max_len);
        fifo_reset_all();
        arducam_camlock_give();
        return;
    }

    // Read out the frame (your existing reader). If you have an
    // “exact until EOI” reader, use it here instead.
    e = arducam_read_jpeg(jpg, max_len, &actual);

    // We’re done with camera HW → release the lock ASAP
    arducam_camlock_give();
    // ---- END CRITICAL SECTION ----

    ESP_LOGI("cam","read_jpeg: %s, len=%u", esp_err_to_name(e), (unsigned)actual);

    if (e == ESP_OK && actual >= 4 &&
        jpg[0] == 0xFF && jpg[1] == 0xD8 &&
        jpg[actual-2] == 0xFF && jpg[actual-1] == 0xD9)
    {
        // Decode outside the lock (heavy)
        uint16_t *pix = NULL; uint16_t w=0, h=0;
        esp_err_t d = jpeg_decode_from_buffer(jpg, actual, &pix, &w, &h, JPEG_IMAGE_SCALE_1);
        if (d == ESP_OK) {
            ESP_LOGI("main","decoded %ux%u", w, h);

            uint8_t *gray = (uint8_t*)heap_caps_malloc((size_t)w*h, MALLOC_CAP_8BIT);
            if (gray) {
                rgb565_to_gray8_with_probe(pix, gray, w, h);
                wifi_cam_publish_gray8_as_bmp(gray, w, h);
                free(gray);
            } else {
                ESP_LOGW("main","OOM gray %u bytes", (unsigned)((size_t)w*h));
            }
            free(pix);
        } else {
            ESP_LOGW("main","decode failed: %s", esp_err_to_name(d));
            // (Optional) publish JPEG preview:
            // wifi_cam_publish_jpeg(jpg, actual);
        }
    } else {
        ESP_LOGW("cam","EOI not found — dropping frame");
        // (Optional) you can still publish partial JPEG for debugging
    }

    free(jpg);

    // Give Wi-Fi/httpd some breathing room between frames
    vTaskDelay(pdMS_TO_TICKS(10));
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
    // (Most OV2640 init sequences use 0xFF=0x01 for ID reads)
    if (i2c_write_reg(0xFF, 0x01) != ESP_OK) {
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

struct camera_operate arducam = {
    .slave_address = 0x30,
    .systemInit    = esp32c3_SystemInit, // Point to your new ESP32 init function
    .busDetect     = spiBusDetect,
    .cameraProbe   = ov2640Probe,
    .cameraInit    = ov2640Init,
    .setJpegSize   = OV2640_set_JPEG_size,
};
