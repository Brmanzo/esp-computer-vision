
#include <stdio.h>
#include <string.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/i2c_master.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_rom_sys.h"
#include "driver/uart.h"

#include "arducam.h"
#include "ov2640.h"
#include "wifi_cam.h"
#include "esp_http_server.h"

#define BANK_SEL   0xFF
#define BANK_DSP   0x00
#define IMAGE_MODE 0xDA
#define R_BYPASS   0x05
#define R_DVP_SP   0xD3

// ArduChip Register Addresses
#define ARDUCHIP_TEST1              0x00  // Test Register
#define ARDUCHIP_FIFO               0x04  // FIFO Control Register
#define ARDUCHIP_GPIO_WRITE_REG     0x06  // GPIO Write Register (for sensor power/reset)
#define ARDUCHIP_GPIO_CTRL_REG      0x07  // GPIO Control Register (for CPLD reset)

// Bitmasks for GPIO Control Register (0x07)
#define GPIO_CPLD_RESET_MASK 0x80 // Bit to reset the CPLD/ArduChip

// Bitmasks for GPIO Write Register (0x06)
#define GPIO_PWREN_MASK      0x04 // 0 = Sensor LDO disable, 1 = sensor LDO enable [1, 2]
#define GPIO_PWDN_MASK       0x02 // 0 = Sensor normal operation, 1 = Sensor standby (power down) [1, 2]
#define GPIO_RESET_MASK      0x01 // OV2640 sensor reset is active-low

// === SPI-Wrapped I2C Functions ===
// These functions tunnel I2C commands over the SPI bus to the ArduChip.
#define SENSOR_REGISTER_ADDRESS 0x7C
#define SENSOR_DATA             0x7D
#define SENSOR_OPERATION_START  0x7E
#define OPERATION_WAIT_FLAG     0x08

// ArduChip Bitmasks
#define FIFO_CLEAR_MASK     0x01  // Bit to clear the FIFO
#define CAP_DONE_MASK      0x08  // Bit that indicates capture is complete

void set_bit(unsigned char addr, unsigned char bit);
void clear_bit(unsigned char addr, unsigned char bit);

i2c_master_bus_handle_t bus_handle;
i2c_master_dev_handle_t camera_dev_handle;

spi_device_handle_t spi_device_handle;

volatile uint8_t cameraCommand = 0;
static QueueHandle_t uart_queue_handle;
static const char* TAG = "UART_HANDLER";
static uint8_t *dummy_tx;

/* Write to device register using SPI. */
void write_reg(uint8_t address, uint8_t value) {
    uint8_t tx_buffer[2] = {address | WRITE_BIT, value};
    
    spi_transaction_t t = {0};
    t.length = 16; // Length is in bits (2 bytes * 8 bits/byte)
    t.tx_buffer = tx_buffer;
    esp_err_t err = spi_device_polling_transmit(spi_device_handle, &t);
    if (err!= ESP_OK) {
        ESP_LOGE("SPI", "Failed to write to register 0x%02X", address);
    }
}

/* Read from device register using SPI. */
uint8_t read_reg(uint8_t address) {
    uint8_t tx[2] = { (uint8_t)(address & 0x7F), 0x00 }; // bit7=0 => read
    uint8_t rx[2] = {0};
    spi_transaction_t t = {0};
    t.length   = 16;            // total bits
    t.rxlength = 16;            // receive full 2 bytes
    t.tx_buffer = tx;
    t.rx_buffer = rx;
    ESP_ERROR_CHECK(spi_device_polling_transmit(spi_device_handle, &t));
    return rx[1];               // second byte is the data
}

/* Set bit at address using SPI. */
void set_bit(unsigned char addr, unsigned char bit)
{
	unsigned char tmp;
	tmp = read_reg(addr);
	write_reg(addr, tmp | bit);
}
/* Clear bit at address using SPI. */
void clear_bit(unsigned char addr, unsigned char bit)
{
	unsigned char tmp;
	tmp = read_reg(addr);
	write_reg(addr, tmp & (~bit));
}

/* Get bit at address using SPI. */
unsigned char get_bit(unsigned char addr, unsigned char bit)
{
  unsigned char tmp;
  tmp = read_reg(addr);
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

    dev_config.device_address = I2C_SLAVE_ADDR;
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

// /* Initialization sequence for the ArduCAM-M-2MP device. */
// static void arducam_power_up_sensor(void) {
//     // Enable sensor LDO
//     set_bit(ARDUCHIP_GPIO, GPIO_PWREN_MASK);
//     vTaskDelay(pdMS_TO_TICKS(5));

//     // Ensure NOT in power-down
//     clear_bit(ARDUCHIP_GPIO, GPIO_PWDN_MASK);
//     vTaskDelay(pdMS_TO_TICKS(5));

//     // Toggle reset: 0 = reset, 1 = normal operation
//     clear_bit(ARDUCHIP_GPIO, GPIO_RESET_MASK);
//     vTaskDelay(pdMS_TO_TICKS(5));
//     set_bit(ARDUCHIP_GPIO, GPIO_RESET_MASK);
//     vTaskDelay(pdMS_TO_TICKS(10));  // let it come up
// }

int rdSensorReg8_8(uint8_t regID, uint8_t* regDat) {
    esp_err_t err = i2c_master_transmit_receive(
        camera_dev_handle, 
        &regID, 1,          // Write the register address we want to read from
        regDat, 1,          // Read one byte of data back
        pdMS_TO_TICKS(I2C_TIMEOUT_MS)
    );

    if (err!= ESP_OK) {
        ESP_LOGE("SCCB", "Read from reg 0x%02X failed: %s", regID, esp_err_to_name(err));
    }
    return err;
}


int wrSensorReg8_8(uint8_t regID, uint8_t regDat) {
    uint8_t buf[2] = {regID, regDat};
    esp_err_t err = i2c_master_transmit(camera_dev_handle, buf, 2, pdMS_TO_TICKS(I2C_TIMEOUT_MS));
    
    if (err!= ESP_OK) {
        ESP_LOGE("SCCB", "Write to reg 0x%02X failed: %s", regID, esp_err_to_name(err));
    }
    return err;
}

/* Write multiple 8-bit values to 8-bit registers on the sensor. */
int wrSensorRegs8_8(const struct sensor_reg *reglist) {
    int err = 0;
    const struct sensor_reg *next = reglist;
    while (next->reg!= 0xff || next->val!= 0xff) {
        err = wrSensorReg8_8(next->reg, next->val);
        if (err!= 0) {
            return err;
        }
        next++;
    }
    return 0;
}

static inline void bank_dsp(void){ wrSensorReg8_8(0xFF, 0x00); }  // BANK_DSP
static inline void bank_sensor(void){ wrSensorReg8_8(0xFF, 0x01); }

static inline esp_err_t bank(uint8_t b){
    esp_err_t e = wrSensorReg8_8(0xFF, b);
    vTaskDelay(pdMS_TO_TICKS(1));   // small settle
    return e;
}

static void ov2640_force_yuv422_strict(void)
{
    bank(0x00);                    // DSP
    wrSensorReg8_8(0xE0, 0x14);    // reset JPEG+DVP while changing
    wrSensorReg8_8(0xDA, 0x00);    // JPEG_EN=0, Y-first (YUYV)
    wrSensorReg8_8(0xD7, 0x03);
    wrSensorReg8_8(0xE1, 0x67);
    wrSensorReg8_8(0x05, 0x01);    // R_BYPASS: DSP enable
    wrSensorReg8_8(0xD3, 0x04);    // very slow PCLK div
    uint8_t c2=0; if (rdSensorReg8_8(0xC2,&c2)==ESP_OK) wrSensorReg8_8(0xC2, (uint8_t)(c2 & ~(1u<<4)));
    wrSensorReg8_8(0xE0, 0x00);
    vTaskDelay(pdMS_TO_TICKS(10));
}

static void sccb_verify(void)
{
    uint8_t v1=0, v2=0, v3=0, v4=0;
    wrSensorReg8_8(0xFF, 0x00);
    rdSensorReg8_8(0xDA, &v1);    // IMAGE_MODE
    rdSensorReg8_8(0xC2, &v2);    // CTRL2
    rdSensorReg8_8(0x05, &v3);    // R_BYPASS
    rdSensorReg8_8(0xD3, &v4);    // R_DVP_SP
    ESP_LOGI("OV2640","readback DA=%02X C2=%02X 05=%02X D3=%02X", v1, v2, v3, v4);
}

static void ov2640_dump_core_regs(void)
{
    uint8_t da=0, c2=0, r05=0, d3=0;
    bank_dsp();
    if (rdSensorReg8_8(0xDA, &da) != ESP_OK ||
        rdSensorReg8_8(0xC2, &c2) != ESP_OK ||
        rdSensorReg8_8(0x05, &r05) != ESP_OK ||
        rdSensorReg8_8(0xD3, &d3) != ESP_OK) {
        ESP_LOGE("OV2640","SCCB read failed (DA/C2/05/D3)");
        return;
    }
    ESP_LOGI("OV2640","DA=0x%02X C2=0x%02X 05=0x%02X D3=0x%02X", da,c2,r05,d3);
    ESP_LOGI("OV2640","Expect: JPEG_EN(DA[4])=0, Y-first(DA[0])=0, C2[4]=0, R_BYPASS(05)=1");
}

/* Read a byte from the SPI FIFO buffer. */
unsigned char read_fifo(void)
{
	unsigned char data;
	data = read_reg(SINGLE_FIFO_READ);
	return data;
}

/* Flush the SPI FIFO buffer. */
void flush_fifo(void)
{
	write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
}

/* Clear the SPI FIFO flag. */
void clear_fifo_flag(void )
{
    write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
}

/* Start the image capture. */
void start_capture(void)
{
	write_reg(ARDUCHIP_FIFO, FIFO_START_MASK);
}

/* Ensure there is a dummy transaction buffer available for FIFO burst. */
static inline void ensure_dummy(void) {
    if (!dummy_tx) {
        dummy_tx = heap_caps_malloc(SPI_CHUNK, MALLOC_CAP_DMA);
        assert(dummy_tx);
        memset(dummy_tx, 0, SPI_CHUNK);
    }
}
/* Burst-read data from the SPI FIFO buffer into a provided buffer. */
void set_fifo_burst(uint8_t *buffer, uint32_t length)
{
    ensure_dummy();
    ESP_ERROR_CHECK(spi_device_acquire_bus(spi_device_handle, portMAX_DELAY));

    uint8_t cmd = BURST_FIFO_READ;          // 0x3C
    spi_transaction_t t0 = {0};
    t0.length = 8;
    t0.tx_buffer = &cmd;
    t0.flags = SPI_TRANS_CS_KEEP_ACTIVE;
    ESP_ERROR_CHECK(spi_device_polling_transmit(spi_device_handle, &t0));

    // *** Discard 1 dummy byte after entering burst ***
    uint8_t trash = 0;
    spi_transaction_t td = {0};
    td.length   = 8;
    td.rxlength = 8;
    td.tx_buffer = dummy_tx;                // any byte
    td.rx_buffer = &trash;
    td.flags = SPI_TRANS_CS_KEEP_ACTIVE;
    ESP_ERROR_CHECK(spi_device_polling_transmit(spi_device_handle, &td));

    uint8_t trash2=0;
    spi_transaction_t td2 = { .length=8, .rxlength=8, .tx_buffer=dummy_tx, .rx_buffer=&trash2, .flags=SPI_TRANS_CS_KEEP_ACTIVE };
    ESP_ERROR_CHECK(spi_device_polling_transmit(spi_device_handle, &td2));

    // Now read the real payload
    while (length > 0) {
        uint32_t n = (length > SPI_CHUNK) ? SPI_CHUNK : length;
        spi_transaction_t t = {0};
        t.length   = n * 8;
        t.rxlength = n * 8;
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
    len1 = read_reg(FIFO_SIZE1);
    len2 = read_reg(FIFO_SIZE2);
    len3 = read_reg(FIFO_SIZE3) & 0x7f;
    len = ((len3 << 16) | (len2 << 8) | len1) & 0x07fffff;
	return len;	
}

static void ov2640_dump_state(void)
{
    ESP_LOGI(TAG, "--- Dumping OV2640 Register State ---");

    // --- SENSOR BANK (0x01) ---
    wrSensorReg8_8(0xFF, 0x01);
    uint8_t val;
    ESP_LOGI(TAG, "--- SENSOR BANK ---");
    bank(0x01); rdSensorReg8_8(0x0A, &val); ESP_LOGI(TAG, "PID [0x0A]: 0x%02X", val);
    bank(0x01); rdSensorReg8_8(0x0B, &val); ESP_LOGI(TAG, "VER: 0x%02X", val);
    bank(0x01); rdSensorReg8_8(0x11, &val); ESP_LOGI(TAG, "CLKRC [0x11]: 0x%02X", val);
    bank(0x01); rdSensorReg8_8(0x12, &val); ESP_LOGI(TAG, "COM7 [0x12]: 0x%02X", val);
    bank(0x01); rdSensorReg8_8(0x03, &val); ESP_LOGI(TAG, "COM1 [0x03]: 0x%02X", val);
    bank(0x01); rdSensorReg8_8(0x04, &val); ESP_LOGI(TAG, "COM1 [0x04]: 0x%02X", val);
    bank(0x01); rdSensorReg8_8(0x17, &val); ESP_LOGI(TAG, "HSTART [0x17]: 0x%02X", val);
    bank(0x01); rdSensorReg8_8(0x18, &val); ESP_LOGI(TAG, "HSIZE [0x18]: 0x%02X", val);
    bank(0x01); rdSensorReg8_8(0x19, &val); ESP_LOGI(TAG, "VSTART [0x19]: 0x%02X", val);
    bank(0x01); rdSensorReg8_8(0x1A, &val); ESP_LOGI(TAG, "VSIZE [0x1A]: 0x%02X", val);

    // --- DSP BANK (0x00) ---
    wrSensorReg8_8(0xFF, 0x00);
    ESP_LOGI(TAG, "--- DSP BANK ---");
    bank(0x00); rdSensorReg8_8(0xDA, &val); ESP_LOGI(TAG, "IMAGE_MODE: 0x%02X", val);
    bank(0x00); rdSensorReg8_8(0xD3, &val); ESP_LOGI(TAG, "R_DVP_SP: 0x%02X", val);
    bank(0x00); rdSensorReg8_8(0xC2, &val); ESP_LOGI(TAG, "CTRL2 [0xC2]: 0x%02X", val);
    bank(0x00); rdSensorReg8_8(0x05, &val); ESP_LOGI(TAG, "R_BYPASS [0x05]: 0x%02X", val);
    bank(0x00); rdSensorReg8_8(0x5A, &val); ESP_LOGI(TAG, "Y_SIZE [0x5A]: 0x%02X", val);
    bank(0x00); rdSensorReg8_8(0x5B, &val); ESP_LOGI(TAG, "X_SIZE: 0x%02X", val);


    uint8_t v;
    esp_err_t e;
    bank(0x00);
    e = rdSensorReg8_8(0xDA,&v);
    ESP_LOGI("OV2640","rd DA e=%d v=%02X", (int)e, v);
    ESP_LOGI(TAG, "--- End of Dump ---");
}

static inline void fifo_reset_all(void) {
    // Clear + reset read/write pointers
    write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK | FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
}

/* Perform a single image capture and publish to the esp's server.*/
void singleCapture(void) {
    // Prepare FIFO and start capture
    fifo_reset_all();
    start_capture();

    // Wait for capture done
    while (!get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
        vTaskDelay(pdMS_TO_TICKS(1));
    }

    // Read JPEG length and validate
    uint32_t length = read_fifo_length();
    if (length == 0 || length > WIFI_CAM_MAX_JPEG) {
        ESP_LOGE("cam", "Bad FIFO length: %lu (max %u)", (unsigned long)length, WIFI_CAM_MAX_JPEG);
        return;
    }

    // Allocate exactly what we need (DMA-capable is fine)
    uint8_t *image_buffer = heap_caps_malloc(length, MALLOC_CAP_DMA);
    if (!image_buffer) {
        ESP_LOGE("cam", "No memory for %lu bytes", (unsigned long)length);
        return;
    }

    // Burst-read the JPEG into RAM
    set_fifo_burst(image_buffer, length);

    // (Optional) quick sanity check of JPEG markers
    bool soi = (length >= 2 && image_buffer[0] == 0xFF && image_buffer[1] == 0xD8);
    bool eoi = (length >= 2 && image_buffer[length-2] == 0xFF && image_buffer[length-1] == 0xD9);
    ESP_LOGI("cam", "JPEG len=%lu SOI=%d EOI=%d", (unsigned long)length, soi, eoi);

    // 6) Publish to the web handler and free
    wifi_cam_publish(image_buffer, length);
    free(image_buffer);
}

static void camera_preroll(int frames)
{
    for (int i=0; i<frames; ++i) {
        fifo_reset_all();
        start_capture();
        while (!get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) { vTaskDelay(pdMS_TO_TICKS(1)); }
        // Just reset again; no need to read out
        fifo_reset_all();
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}
/* Prepares ESP for communication and initiates image capture. */
// This single function performs the entire initialization sequence correctly.
void arducam_yuv_init(void) {
    // 1) Peripherals
    spi_master_init();
    i2c_master_init();     // direct SCCB to OV2640
    uart_init();

    // 2) ArduCHIP GPIO direction, then power sequence
    write_reg(0x05, 0x07);                                 // GPIO dir: RESET/PWDN/PWREN = outputs
    write_reg(0x06, read_reg(0x06) | 0x04);                // PWREN=1
    vTaskDelay(pdMS_TO_TICKS(5));
    write_reg(0x06, read_reg(0x06) & ~0x02);               // PWDN=0
    vTaskDelay(pdMS_TO_TICKS(5));
    write_reg(0x06, read_reg(0x06) & ~0x01);               // RESET=0 (active-low)
    vTaskDelay(pdMS_TO_TICKS(5));
    write_reg(0x06, read_reg(0x06) | 0x01);                // RESET=1 (release)
    vTaskDelay(pdMS_TO_TICKS(10));

    // 3) SPI sanity: Test register should echo 0x55
    write_reg(0x03, 0x00); // Set timing
    write_reg(0x00, 0x55);
    uint8_t t = read_reg(0x00);
    if (t != 0x55) { ESP_LOGE("ARDUCHIP","SPI test failed: 0x%02X", t); return; }

    // 4) Sensor ID over I²C (SCCB)
    write_reg(ARDUCHIP_MODE, MCU2LCD_MODE);
    wrSensorReg8_8(0xFF, 0x01); // BANK_SENSOR
    uint8_t idh=0, idl=0;
    rdSensorReg8_8(0x0A, &idh);
    rdSensorReg8_8(0x0B, &idl);
    ESP_LOGI("OV2640","ID=%02X%02X", idh, idl);
    // Expect 26 40/41/42; if not, fix I2C address/wiring/pullups
    ov2640_dump_state();
    // 5) Sensor init for YUV (direct I²C)
    // (Your ov2640_yuv_qvga_init_regs[] is fine; it’s a typical YUV+QVGA base)
    wrSensorReg8_8(0xFF, 0x01);
    wrSensorReg8_8(0x12, 0x80);            // reset
    vTaskDelay(pdMS_TO_TICKS(10));
    wrSensorRegs8_8(ov2640_yuv_qvga_init_regs);
    vTaskDelay(pdMS_TO_TICKS(10));

    // 6) (Optional) assert DSP path & easy PCLK on DSP bank
    wrSensorReg8_8(0xFF, 0x00);            // BANK_DSP
    wrSensorReg8_8(0x05, 0x01);            // R_BYPASS: DSP enable
    wrSensorReg8_8(0xD3, 0x82);            // R_DVP_SP: small divider

    // 7) Force YUV422 mode (just in case)
    ov2640_force_yuv422_strict();
    sccb_verify();
    ov2640_dump_core_regs();
    camera_preroll(2);

    ESP_LOGI("OV2640","YUV init done.");
}


/* Detect the SPI bus operational state. */
uint8_t spiBusDetect(void){
    write_reg(0x00, 0x55);
    if(read_reg(0x00) == 0x55){
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
    if (wrSensorReg8_8(0xFF, 0x01) != ESP_OK) {
        printf("OV2640: failed to select ID bank\r\n");
        return 1;
    }

    if (rdSensorReg8_8(0x0A, &id_H) != ESP_OK ||
        rdSensorReg8_8(0x0B, &id_L) != ESP_OK) {
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

// --- little-endian writers for BMP header ---
static inline void put_le16(uint8_t *p, uint16_t v){ p[0]=v&0xFF; p[1]=v>>8; }
static inline void put_le32(uint8_t *p, uint32_t v){
    p[0]=v&0xFF; p[1]=(v>>8)&0xFF; p[2]=(v>>16)&0xFF; p[3]=(v>>24)&0xFF;
}

static SemaphoreHandle_t s_cam_mutex;

void arducam_camlock_take(void){
    if (!s_cam_mutex) s_cam_mutex = xSemaphoreCreateMutex();
    xSemaphoreTake(s_cam_mutex, portMAX_DELAY);
}
void arducam_camlock_give(void){
    if (s_cam_mutex) xSemaphoreGive(s_cam_mutex);
}

#define TAG_OV "OV2640"
#define CHUNK  4092  // keep even

static uint8_t s_chunk[CHUNK];
static uint8_t s_yrow[640];   // up to VGA width

// little-endian writers for BMP header
static inline void le16(uint8_t *p, uint16_t v){ p[0]=v; p[1]=v>>8; }
static inline void le32(uint8_t *p, uint32_t v){ p[0]=v; p[1]=v>>8; p[2]=v>>16; p[3]=v>>24; }

/**
 * Stream a single grayscale BMP built from the Y component of a YUV422 frame.
 * Assumes sensor is configured for YUV422 and ArduCAM FIFO contains W*H*2 bytes.
 */
// Return 0 if even bytes look like Y, 1 if odd bytes look like Y
static int autodetect_y_index(const uint8_t *buf, size_t n)
{
    if (n < 512) return 0; // default
    // compute variance of even vs odd
    double sumE=0, sumO=0, sum2E=0, sum2O=0; size_t cntE=0, cntO=0;
    // sample first ~2 KB for speed
    size_t limit = n < 2048 ? n : 2048;
    // only look at pairs to avoid crossing chunk oddities
    for (size_t i=0; i+1<limit; i+=2) {
        uint8_t e = buf[i], o = buf[i+1];
        sumE += e; sum2E += (double)e*e; cntE++;
        sumO += o; sum2O += (double)o*o; cntO++;
    }
    double varE = (sum2E / (cntE?cntE:1)) - (sumE/cntE)*(sumE/cntE);
    double varO = (sum2O / (cntO?cntO:1)) - (sumO/cntO)*(sumO/cntO);
    // Y has larger variance than chroma (which hovers near ~128)
    return (varO > varE) ? 1 : 0;
}

esp_err_t arducam_stream_gray_bmp(httpd_req_t *req, uint16_t W, uint16_t H)
{
    if (W > sizeof s_yrow) return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "frame too wide");

    arducam_camlock_take();

    // 1) Capture into FIFO
    fifo_reset_all();
    start_capture();
    while (!get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) { vTaskDelay(pdMS_TO_TICKS(1)); }

    uint32_t fifo_len = read_fifo_length();
    uint32_t expected = (uint32_t)W * H * 2;
    if (fifo_len != expected) {
        // Normal for YUV (size regs are JPEG-oriented) — we’ll read 'expected'
        ESP_LOGW("OV2640","fifo_len=%u expected=%u (ignoring size regs for YUV)",
                 (unsigned)fifo_len, (unsigned)expected);
    }

    // ALWAYS read the expected amount for YUV:
    size_t remaining = expected;

    // 2) HTTP headers (8-bit top-down BMP)
    httpd_resp_set_type(req, "image/bmp");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");

    const uint32_t offBits   = 14 + 40 + 1024;
    const uint32_t img_bytes = (uint32_t)W * H;
    const uint32_t file_size = offBits + img_bytes;

    uint8_t hdr[14+40] = {0};
    hdr[0]='B'; hdr[1]='M';
    le32(&hdr[ 2], file_size);
    le32(&hdr[10], offBits);
    le32(&hdr[14], 40);
    le32(&hdr[18], W);
    le32(&hdr[22], (uint32_t)(-(int32_t)H));   // top-down
    le16(&hdr[26], 1);
    le16(&hdr[28], 8);
    le32(&hdr[34], img_bytes);
    ESP_ERROR_CHECK(httpd_resp_send_chunk(req, (const char*)hdr, sizeof hdr));

    // grayscale palette
    uint8_t pal[1024];
    for (int i=0;i<256;i++){ pal[i*4+0]=i; pal[i*4+1]=i; pal[i*4+2]=i; pal[i*4+3]=0; }
    ESP_ERROR_CHECK(httpd_resp_send_chunk(req, (const char*)pal, sizeof pal));

    // 3) Begin BURST read and stream rows
    ensure_dummy();
    ESP_ERROR_CHECK(spi_device_acquire_bus(spi_device_handle, portMAX_DELAY));
    uint8_t cmd = BURST_FIFO_READ;
    spi_transaction_t t0 = { .length=8, .tx_buffer=&cmd, .flags=SPI_TRANS_CS_KEEP_ACTIVE };
    ESP_ERROR_CHECK(spi_device_polling_transmit(spi_device_handle, &t0));

    size_t y_fill = 0;
    int y_index = 0;
    bool decided = false;
    bool dumped  = false;

    while (remaining) {
        size_t n = remaining > CHUNK ? CHUNK : remaining;

        spi_transaction_t t = {0};
        t.length   = n * 8;
        t.rxlength = n * 8;
        t.tx_buffer= dummy_tx;
        t.rx_buffer= s_chunk;
        t.flags    = (n == remaining) ? 0 : SPI_TRANS_CS_KEEP_ACTIVE;
        ESP_ERROR_CHECK(spi_device_polling_transmit(spi_device_handle, &t));

        if (!decided) {
            y_index = autodetect_y_index(s_chunk, n);
            ESP_LOGI("OV2640", "Y auto-detect: using %s bytes as Y", y_index==0 ? "even" : "odd");
            decided = true;
        }

        // Extract Y from this chunk
        for (size_t i = y_index; i + 1 < n; i += 2) {
            s_yrow[y_fill++] = s_chunk[i];     // if UYVY, use s_chunk[i+1]
            if (y_fill == W) {
                ESP_ERROR_CHECK(httpd_resp_send_chunk(req, (const char*)s_yrow, W));
                y_fill = 0;
            }
        }

        // Optional: dump first 32 bytes only for debug
        if (!dumped) {
            for (int k=0; k<32 && k<(int)n; k++) {
                printf("%02X%s", s_chunk[k], ((k&15)==15) ? "\n" : " ");
            }
            dumped = true;
        }

        remaining -= n;
    }
    spi_device_release_bus(spi_device_handle);

    if (y_fill) { // flush partial row (shouldn’t happen if expected is correct)
        memset(s_yrow + y_fill, s_yrow[y_fill-1], W - y_fill);
        ESP_ERROR_CHECK(httpd_resp_send_chunk(req, (const char*)s_yrow, W));
    }

    ESP_ERROR_CHECK(httpd_resp_send_chunk(req, NULL, 0));
    arducam_camlock_give();
    return ESP_OK;
}