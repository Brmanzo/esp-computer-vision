
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
#include "ov2640_settings.h"
#include "sensor.h"

void set_bit(unsigned char addr, unsigned char bit);
void clear_bit(unsigned char addr, unsigned char bit);

i2c_master_bus_handle_t bus_handle;
i2c_master_dev_handle_t camera_dev_handle;

spi_device_handle_t spi_device_handle;

volatile uint8_t cameraCommand = 0;
static QueueHandle_t uart_queue_handle;
static const char* TAG = "UART_HANDLER";
static uint8_t *dummy_tx;

#define BANK_SENSOR 1
#define COM7 0x12
#define COM7_SRST 0x80
static uint8_t sccb_current_bank = 0xFF;  // invalid at boot

/* Write to device register using SPI. */
void write_reg(uint8_t address, uint8_t value) {
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
uint8_t read_reg(uint8_t address) {
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
    set_bit(ARDUCHIP_GPIO, GPIO_PWREN_MASK);
    vTaskDelay(pdMS_TO_TICKS(5));

    // Ensure NOT in power-down
    clear_bit(ARDUCHIP_GPIO, GPIO_PWDN_MASK);
    vTaskDelay(pdMS_TO_TICKS(5));

    // Toggle reset: 0 = reset, 1 = normal operation
    clear_bit(ARDUCHIP_GPIO, GPIO_RESET_MASK);
    vTaskDelay(pdMS_TO_TICKS(5));
    set_bit(ARDUCHIP_GPIO, GPIO_RESET_MASK);
    vTaskDelay(pdMS_TO_TICKS(10));  // let it come up
}

/* Prepares ESP for communication and initiates image capture. */
void esp32c3_SystemInit(void) {
    i2c_master_init();
    spi_master_init();
    uart_init();
    arducam_power_up_sensor();
}

/* Write an 8-bit value to an 8-bit register on the sensor. */
int wrSensorReg8_8(uint8_t regID, uint8_t regDat) {
    uint8_t buf[2] = {regID, regDat};
    esp_err_t err = i2c_master_transmit(camera_dev_handle, buf, 2, pdMS_TO_TICKS(I2C_TIMEOUT_MS));
    if (err != ESP_OK) {
        ESP_LOGE("SCCB", "Write reg 0x%02X=0x%02X failed: %s", regID, regDat, esp_err_to_name(err));
    }
    return err;
}

/* Switch the sensor register bank if needed. */
static inline int wrBank(uint8_t bank) {
    if (sccb_current_bank == bank) return 0;
    int r = wrSensorReg8_8(BANK_SEL, bank);
    if (!r) { sccb_current_bank = bank; vTaskDelay(pdMS_TO_TICKS(1)); } // small settle
    return r;
}

static inline int wrSensorReg8_8_banked(uint8_t bank, uint8_t reg, uint8_t val) {
    int r = wrBank(bank);
    if (r) return r;
    r = wrSensorReg8_8(reg, val);
    if (!r && reg == COM7 && (val & COM7_SRST)) vTaskDelay(pdMS_TO_TICKS(10)); // reset settle
    return r;
}

/* Write multiple 8-bit values to 8-bit registers on the sensor. */
int wrSensorRegs8_8(const sensor_reg *reglist) {
    for (const sensor_reg *p = reglist; !(p->reg == 0xFF && p->val == 0xFF); ++p) {
        int err;
        if (p->reg == BANK_SEL) {
            err = wrBank(p->val);  // updates cache + small delay
        } else {
            err = wrSensorReg8_8(p->reg, p->val);
            if (!err && p->reg == COM7 && (p->val & COM7_SRST)) vTaskDelay(pdMS_TO_TICKS(10));
        }
        if (err) return err;
    }
    return 0;
}

static inline esp_err_t rdSensorReg8_8_banked(uint8_t bank, uint8_t regID, uint8_t *out)
{
    esp_err_t r = wrBank(bank);
    if (r != ESP_OK) return r;
    return rdSensorReg8_8(regID, out);
}

// get_reg_bits(): return status; result in *out_bits
static inline int rdSensorRegs8_8(uint8_t bank, uint8_t reg,
                                     uint8_t offset, uint8_t mask)
{
    uint8_t v = 0;
    if (rdSensorReg8_8_banked(bank, reg, &v) != ESP_OK) return -1;
    return (int)((v >> offset) & mask);
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

    // 2) Read all data in 4092-byte chunks; keep CS low until last
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
    len1 = read_reg(FIFO_SIZE1);
    len2 = read_reg(FIFO_SIZE2);
    len3 = read_reg(FIFO_SIZE3) & 0x7f;
    len = ((len3 << 16) | (len2 << 8) | len1) & 0x07fffff;
	return len;	
}
/* Set the JPEG size for the OV2640 camera. */
void OV2640_set_JPEG_size(unsigned char size)
{
	switch(size)
	{
		case res_160x120:
			wrSensorRegs8_8(OV2640_160x120_JPEG);
			break;
		case res_176x144:
			wrSensorRegs8_8(OV2640_176x144_JPEG);
			break;
		case res_320x240:
			wrSensorRegs8_8(OV2640_320x240_JPEG);
			break;
		case res_352x288:
	  	wrSensorRegs8_8(OV2640_352x288_JPEG);
			break;
		case res_640x480:
			wrSensorRegs8_8(OV2640_640x480_JPEG);
			break;
		case res_800x600:
			wrSensorRegs8_8(OV2640_800x600_JPEG);
			break;
		case res_1024x768:
			wrSensorRegs8_8(OV2640_1024x768_JPEG);
			break;
		case res_1280x1024:
			wrSensorRegs8_8(OV2640_1280x1024_JPEG);
			break;
		case res_1600x1200:
			wrSensorRegs8_8(OV2640_1600x1200_JPEG);
			break;
		default:
			wrSensorRegs8_8(OV2640_320x240_JPEG);
			break;
	}
}
/* Initialize the OV2640 camera sensor. */
void ov2640Init(){
    wrSensorReg8_8(0xff, 0x01);
    wrSensorReg8_8(0x12, 0x80);
    wrSensorRegs8_8(OV2640_JPEG_INIT);
    wrSensorRegs8_8(OV2640_YUV422);
    wrSensorRegs8_8(OV2640_JPEG);
    wrSensorReg8_8(0xff, 0x01);
    wrSensorReg8_8(0x15, 0x00);
    wrSensorRegs8_8(OV2640_320x240_JPEG);
}

#define WRITE_REG_OR_RETURN(bank, reg, val)          \
    do { int _r = wrSensorReg8_8_banked((bank), (reg), (val)); \
         if (_r) return _r; } while (0)

#define WRITE_REGS_OR_RETURN(regs)                   \
    do { int _r = wrSensorRegs8_8(regs);             \
         if (_r) return _r; } while (0)

#define READ_REG_OR_RETURN(bank, reg, pOut)      \
    do { esp_err_t _r = rdSensorReg8_8_banked((bank), (reg), (pOut)); \
         if (_r != ESP_OK) return _r; } while (0)

/* Initialize the OV2640 using espcam library. */
static int reset(sensor_t *sensor) {
    int ret = 0;
    WRITE_REG_OR_RETURN(BANK_SENSOR, COM7, COM7_SRST); // Reset all registers
    vTaskDelay(10 / portTICK_PERIOD_MS);
    WRITE_REGS_OR_RETURN(ov2640_settings_cif);
    return ret;
}

static int init_status(sensor_t *sensor){
    sensor->status.brightness = 0;
    sensor->status.contrast = 0;
    sensor->status.saturation = 0;
    sensor->status.ae_level = 0;
    sensor->status.special_effect = 0;
    sensor->status.wb_mode = 0;

    sensor->status.agc_gain = 30;
    uint8_t gain_byte = 0;
    ESP_ERROR_CHECK(rdSensorReg8_8_banked(BANK_SENSOR, GAIN, &gain_byte));
    int agc_gain = (int)gain_byte;

    for (int i = 0; i < 30; i++) {
        if (agc_gain >= agc_gain_tbl[i] && agc_gain < agc_gain_tbl[i+1]) {
            sensor->status.agc_gain = i;
            break;
        }
    }
    int reg45_5_0 = rdSensorRegs8_8(BANK_SENSOR, REG45, 0, 0x3F);   // bits [5:0]

    uint8_t aec_byte = 0;
    esp_err_t aec_err = rdSensorReg8_8_banked(BANK_SENSOR, AEC, &aec_byte);  // <-- &aec_byte!

    int reg04_1_0 = rdSensorRegs8_8(BANK_SENSOR, REG04, 0, 0x03);   // bits [1:0]

    // error handling: check ints for -1 and the esp_err_t for non-OK
    if (reg45_5_0 < 0 || reg04_1_0 < 0 || aec_err != ESP_OK) {
        ESP_LOGE("OV2640", "AEC read failed");
        return ESP_FAIL;
    }

    sensor->status.aec_value =
        ((uint16_t)reg45_5_0 << 10) |
        ((uint16_t)aec_byte  <<  2) |
        (uint16_t)reg04_1_0;   // 0..1200

    uint8_t quality_byte = 0;
    rdSensorReg8_8_banked(BANK_DSP, QS, &quality_byte);
    sensor->status.quality = quality_byte;
    sensor->status.gainceiling = rdSensorRegs8_8(BANK_SENSOR, COM9, 5, 7);
    sensor->status.awb = rdSensorRegs8_8(BANK_DSP, CTRL1, 3, 1);
    sensor->status.awb_gain = rdSensorRegs8_8(BANK_DSP, CTRL1, 2, 1);
    sensor->status.aec = rdSensorRegs8_8(BANK_SENSOR, COM8, 0, 1);
    sensor->status.aec2 = rdSensorRegs8_8(BANK_DSP, CTRL0, 6, 1);
    sensor->status.agc = rdSensorRegs8_8(BANK_SENSOR, COM8, 2, 1);
    sensor->status.bpc = rdSensorRegs8_8(BANK_DSP, CTRL3, 7, 1);
    sensor->status.wpc = rdSensorRegs8_8(BANK_DSP, CTRL3, 6, 1);
    sensor->status.raw_gma = rdSensorRegs8_8(BANK_DSP, CTRL1, 5, 1);
    sensor->status.lenc = rdSensorRegs8_8(BANK_DSP, CTRL1, 1, 1);
    sensor->status.hmirror = rdSensorRegs8_8(BANK_SENSOR, REG04, 7, 1);
    sensor->status.vflip = rdSensorRegs8_8(BANK_SENSOR, REG04, 6, 1);
    sensor->status.dcw = rdSensorRegs8_8(BANK_DSP, CTRL2, 5, 1);
    sensor->status.colorbar = rdSensorRegs8_8(BANK_SENSOR, COM7, 1, 1);

    sensor->status.sharpness = 0;//not supported
    sensor->status.denoise = 0;
    return 0;
}

static int set_pixformat(sensor_t *sensor, pixformat_t pixformat)
{
    int ret = 0;
    sensor->pixformat = pixformat;
    switch (pixformat) {
    case PIXFORMAT_RGB565:
    case PIXFORMAT_RGB888:
        WRITE_REGS_OR_RETURN(ov2640_settings_rgb565);
        break;
    case PIXFORMAT_YUV422:
    case PIXFORMAT_GRAYSCALE:
        WRITE_REGS_OR_RETURN(ov2640_settings_yuv422);
        break;
    case PIXFORMAT_JPEG:
        WRITE_REGS_OR_RETURN(ov2640_settings_jpeg3);
        break;
    default:
        ret = -1;
        break;
    }
    if(!ret) {
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }

    return ret;
}

static int set_framesize(sensor_t *sensor, framesize_t framesize)
{
    int ret = 0;
    uint16_t w = resolution[framesize].width;
    uint16_t h = resolution[framesize].height;
    aspect_ratio_t ratio = resolution[framesize].aspect_ratio;
    uint16_t max_x = ratio_table[ratio].max_x;
    uint16_t max_y = ratio_table[ratio].max_y;
    uint16_t offset_x = ratio_table[ratio].offset_x;
    uint16_t offset_y = ratio_table[ratio].offset_y;
    ov2640_sensor_mode_t mode = OV2640_MODE_UXGA;

    sensor->status.framesize = framesize;



    if (framesize <= FRAMESIZE_CIF) {
        mode = OV2640_MODE_CIF;
        max_x /= 4;
        max_y /= 4;
        offset_x /= 4;
        offset_y /= 4;
        if(max_y > 296){
            max_y = 296;
        }
    } else if (framesize <= FRAMESIZE_SVGA) {
        mode = OV2640_MODE_SVGA;
        max_x /= 2;
        max_y /= 2;
        offset_x /= 2;
        offset_y /= 2;
    }

    ret = set_window(sensor, mode, offset_x, offset_y, max_x, max_y, w, h);
    return ret;
}

static int set_contrast(sensor_t *sensor, int level)
{
    int ret=0;
    level += 3;
    if (level <= 0 || level > NUM_CONTRAST_LEVELS) {
        return -1;
    }
    sensor->status.contrast = level-3;
    for (int i=0; i<7; i++) {
        WRITE_REG_OR_RETURN(BANK_DSP, contrast_regs[0][i], contrast_regs[level][i]);
    }
    return ret;
}

static int set_brightness(sensor_t *sensor, int level)
{
    int ret=0;
    level += 3;
    if (level <= 0 || level > NUM_BRIGHTNESS_LEVELS) {
        return -1;
    }
    sensor->status.brightness = level-3;
    for (int i=0; i<5; i++) {
        WRITE_REG_OR_RETURN(BANK_DSP, brightness_regs[0][i], brightness_regs[level][i]);
    }
    return ret;
}

static int set_saturation(sensor_t *sensor, int level)
{
    int ret=0;
    level += 3;
    if (level <= 0 || level > NUM_SATURATION_LEVELS) {
        return -1;
    }
    sensor->status.saturation = level-3;
    for (int i=0; i<5; i++) {
        WRITE_REG_OR_RETURN(BANK_DSP, saturation_regs[0][i], saturation_regs[level][i]);
    }
    return ret;
}

static int set_quality(sensor_t *sensor, int quality)
{
    if(quality < 0) {
        quality = 0;
    } else if(quality > 63) {
        quality = 63;
    }
    sensor->status.quality = quality;
    return write_reg(sensor, BANK_DSP, QS, quality);
}

static int set_colorbar(sensor_t *sensor, int enable)
{
    sensor->status.colorbar = enable;
    return write_reg_bits(sensor, BANK_SENSOR, COM7, COM7_COLOR_BAR, enable?1:0);
}

static int set_gainceiling_sensor(sensor_t *sensor, gainceiling_t gainceiling)
{
    sensor->status.gainceiling = gainceiling;
    //return write_reg(sensor, BANK_SENSOR, COM9, COM9_AGC_SET(gainceiling));
    return set_reg_bits(sensor, BANK_SENSOR, COM9, 5, 7, gainceiling);
}

static int set_agc_sensor(sensor_t *sensor, int enable)
{
    sensor->status.agc = enable;
    return write_reg_bits(sensor, BANK_SENSOR, COM8, COM8_AGC_EN, enable?1:0);
}

static int set_aec_sensor(sensor_t *sensor, int enable)
{
    sensor->status.aec = enable;
    return write_reg_bits(sensor, BANK_SENSOR, COM8, COM8_AEC_EN, enable?1:0);
}

static int set_hmirror_sensor(sensor_t *sensor, int enable)
{
    sensor->status.hmirror = enable;
    return write_reg_bits(sensor, BANK_SENSOR, REG04, REG04_HFLIP_IMG, enable?1:0);
}

static int set_vflip_sensor(sensor_t *sensor, int enable)
{
    int ret = 0;
    sensor->status.vflip = enable;
    ret = write_reg_bits(sensor, BANK_SENSOR, REG04, REG04_VREF_EN, enable?1:0);
    return ret & write_reg_bits(sensor, BANK_SENSOR, REG04, REG04_VFLIP_IMG, enable?1:0);
}

static int set_awb_dsp(sensor_t *sensor, int enable)
{
    sensor->status.awb = enable;
    return set_reg_bits(sensor, BANK_DSP, CTRL1, 3, 1, enable?1:0);
}

static int set_aec2(sensor_t *sensor, int enable)
{
    sensor->status.aec2 = enable;
    return set_reg_bits(sensor, BANK_DSP, CTRL0, 6, 1, enable?0:1);
}

static int set_aec_value(sensor_t *sensor, int value)
{
    if(value < 0) {
        value = 0;
    } else if(value > 1200) {
        value = 1200;
    }
    sensor->status.aec_value = value;
    return set_reg_bits(sensor, BANK_SENSOR, REG04, 0, 3, value & 0x3)
           || write_reg(sensor, BANK_SENSOR, AEC, (value >> 2) & 0xFF)
           || set_reg_bits(sensor, BANK_SENSOR, REG45, 0, 0x3F, value >> 10);
}

static int set_special_effect(sensor_t *sensor, int effect)
{
    int ret=0;
    effect++;
    if (effect <= 0 || effect > NUM_SPECIAL_EFFECTS) {
        return -1;
    }
    sensor->status.special_effect = effect-1;
    for (int i=0; i<5; i++) {
        WRITE_REG_OR_RETURN(BANK_DSP, special_effects_regs[0][i], special_effects_regs[effect][i]);
    }
    return ret;
}

static int set_wb_mode(sensor_t *sensor, int mode)
{
    int ret=0;
    if (mode < 0 || mode > NUM_WB_MODES) {
        return -1;
    }
    sensor->status.wb_mode = mode;
    SET_REG_BITS_OR_RETURN(BANK_DSP, 0XC7, 6, 1, mode?1:0);
    if(mode) {
        for (int i=0; i<3; i++) {
            WRITE_REG_OR_RETURN(BANK_DSP, wb_modes_regs[0][i], wb_modes_regs[mode][i]);
        }
    }
    return ret;
}

static int set_ae_level(sensor_t *sensor, int level)
{
    int ret=0;
    level += 3;
    if (level <= 0 || level > NUM_AE_LEVELS) {
        return -1;
    }
    sensor->status.ae_level = level-3;
    for (int i=0; i<3; i++) {
        WRITE_REG_OR_RETURN(BANK_SENSOR, ae_levels_regs[0][i], ae_levels_regs[level][i]);
    }
    return ret;
}

static int set_dcw_dsp(sensor_t *sensor, int enable)
{
    sensor->status.dcw = enable;
    return set_reg_bits(sensor, BANK_DSP, CTRL2, 5, 1, enable?1:0);
}

static int set_bpc_dsp(sensor_t *sensor, int enable)
{
    sensor->status.bpc = enable;
    return set_reg_bits(sensor, BANK_DSP, CTRL3, 7, 1, enable?1:0);
}

static int set_wpc_dsp(sensor_t *sensor, int enable)
{
    sensor->status.wpc = enable;
    return set_reg_bits(sensor, BANK_DSP, CTRL3, 6, 1, enable?1:0);
}

static int set_awb_gain_dsp(sensor_t *sensor, int enable)
{
    sensor->status.awb_gain = enable;
    return set_reg_bits(sensor, BANK_DSP, CTRL1, 2, 1, enable?1:0);
}

static int set_agc_gain(sensor_t *sensor, int gain)
{
    if(gain < 0) {
        gain = 0;
    } else if(gain > 30) {
        gain = 30;
    }
    sensor->status.agc_gain = gain;
    return write_reg(sensor, BANK_SENSOR, GAIN, agc_gain_tbl[gain]);
}

static int set_raw_gma_dsp(sensor_t *sensor, int enable)
{
    sensor->status.raw_gma = enable;
    return set_reg_bits(sensor, BANK_DSP, CTRL1, 5, 1, enable?1:0);
}

static int set_lenc_dsp(sensor_t *sensor, int enable)
{
    sensor->status.lenc = enable;
    return set_reg_bits(sensor, BANK_DSP, CTRL1, 1, 1, enable?1:0);
}

//unsupported
static int set_sharpness(sensor_t *sensor, int level)
{
   return -1;
}

static int set_denoise(sensor_t *sensor, int level)
{
   return -1;
}

static int get_reg(sensor_t *sensor, int reg, int mask)
{
    int ret = read_reg(sensor, (reg >> 8) & 0x01, reg & 0xFF);
    if(ret > 0){
        ret &= mask;
    }
    return ret;
}

static int set_reg(sensor_t *sensor, int reg, int mask, int value)
{
    int ret = 0;
    ret = read_reg(sensor, (reg >> 8) & 0x01, reg & 0xFF);
    if(ret < 0){
        return ret;
    }
    value = (ret & ~mask) | (value & mask);
    ret = write_reg(sensor, (reg >> 8) & 0x01, reg & 0xFF, value);
    return ret;
}

static int set_res_raw(sensor_t *sensor, int startX, int startY, int endX, int endY, int offsetX, int offsetY, int totalX, int totalY, int outputX, int outputY, bool scale, bool binning)
{
    return set_window(sensor, (ov2640_sensor_mode_t)startX, offsetX, offsetY, totalX, totalY, outputX, outputY);
}

static int _set_pll(sensor_t *sensor, int bypass, int multiplier, int sys_div, int root_2x, int pre_div, int seld5, int pclk_manual, int pclk_div)
{
    return -1;
}

static int set_xclk(sensor_t *sensor, int timer, int xclk)
{
    int ret = 0;
    sensor->xclk_freq_hz = xclk * 1000000U;
    ret = xclk_timer_conf(timer, sensor->xclk_freq_hz);
    return ret;
}


int ov2640_init(sensor_t *sensor){
    sensor->reset = reset;
    sensor->init_status = init_status;
    sensor->set_pixformat = set_pixformat;
    sensor->set_framesize = set_framesize;
    sensor->set_contrast  = set_contrast;
    sensor->set_brightness = set_brightness;
    sensor->set_saturation = set_saturation;

    sensor->set_quality = set_quality;
    sensor->set_colorbar = set_colorbar;

    sensor->set_gainceiling = set_gainceiling_sensor;
    sensor->set_gain_ctrl = set_agc_sensor;
    sensor->set_exposure_ctrl = set_aec_sensor;
    sensor->set_hmirror = set_hmirror_sensor;
    sensor->set_vflip = set_vflip_sensor;

    sensor->set_whitebal = set_awb_dsp;
    sensor->set_aec2 = set_aec2;
    sensor->set_aec_value = set_aec_value;
    sensor->set_special_effect = set_special_effect;
    sensor->set_wb_mode = set_wb_mode;
    sensor->set_ae_level = set_ae_level;

    sensor->set_dcw = set_dcw_dsp;
    sensor->set_bpc = set_bpc_dsp;
    sensor->set_wpc = set_wpc_dsp;
    sensor->set_awb_gain = set_awb_gain_dsp;
    sensor->set_agc_gain = set_agc_gain;

    sensor->set_raw_gma = set_raw_gma_dsp;
    sensor->set_lenc = set_lenc_dsp;

    //not supported
    sensor->set_sharpness = set_sharpness;
    sensor->set_denoise = set_denoise;

    sensor->get_reg = get_reg;
    sensor->set_reg = set_reg;
    sensor->set_res_raw = set_res_raw;
    sensor->set_pll = _set_pll;
    sensor->set_xclk = set_xclk;
    ESP_LOGD(TAG, "OV2640 Attached");
    return 0;
}

/* Reset the SPI FIFO buffer. */
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

static sensor_t g_sensor;

void ov2640_attach(void) {
    memset(&g_sensor, 0, sizeof(g_sensor));
    g_sensor.slv_addr = 0x30;      // OV2640 SCCB address
    // g_sensor.xclk_freq_hz = 20000000; // Optional; not needed with ArduCAM clocking
    ov2640_init(&g_sensor);         // <-- fills the function pointers
}

void ov2640_configure_yuv_qvga(void) {
    g_sensor.reset(&g_sensor);                         // runs the reset() you ported (tables like ov2640_settings_cif)
    g_sensor.set_pixformat(&g_sensor, PIXFORMAT_YUV422);
    g_sensor.set_framesize(&g_sensor, FRAMESIZE_QVGA); // 320x240
    // Optional tuning:
    // g_sensor.set_hmirror(&g_sensor, 1);
    // g_sensor.set_vflip(&g_sensor, 0);
    // g_sensor.set_exposure_ctrl(&g_sensor, 1);
}

void ov2640_configure_jpeg_qvga(void) {
    g_sensor.reset(&g_sensor);
    g_sensor.set_pixformat(&g_sensor, PIXFORMAT_JPEG);
    g_sensor.set_framesize(&g_sensor, FRAMESIZE_QVGA);
    g_sensor.set_quality(&g_sensor, 10); // 10..63; lower = better quality/larger
}

struct camera_operate arducam = {
    .slave_address = 0x30,
    .systemInit    = esp32c3_SystemInit, // Point to your new ESP32 init function
    .busDetect     = spiBusDetect,
    .cameraProbe   = ov2640Probe,
    .cameraInit    = ov2640Init,
    .setJpegSize   = OV2640_set_JPEG_size,
};
