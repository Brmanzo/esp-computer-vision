
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
    write_reg(SENSOR_REGISTER_ADDRESS, regID);
    write_reg(SENSOR_OPERATION_START, 0x02); // Start read operation
    while (read_reg(SENSOR_OPERATION_START) & OPERATION_WAIT_FLAG) {
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    *regDat = read_reg(SENSOR_DATA);
    return 0; // Success
}


int wrSensorReg8_8(uint8_t regID, uint8_t regDat) {
    write_reg(SENSOR_REGISTER_ADDRESS, regID);
    write_reg(SENSOR_DATA, regDat);
    write_reg(SENSOR_OPERATION_START, 0x01); // Start write operation
    while (read_reg(SENSOR_OPERATION_START) & OPERATION_WAIT_FLAG) {
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    return 0; // Success
}

/* Write multiple 8-bit values to 8-bit registers on the sensor. */
int wrSensorRegs8_8(const struct sensor_reg reglist[]) {
    int err = 0;
    const struct sensor_reg *next = reglist;
    
    // Check the next entry before processing it.
    while (next->reg != 0xff || next->val != 0xff) {
        // Pass the register and value directly to your write function.
        err = wrSensorReg8_8(next->reg, next->val);
        
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

static void ov2640_dump_state(void)
{
    ESP_LOGI(TAG, "--- Dumping OV2640 Register State ---");

    // --- SENSOR BANK (0x01) ---
    wrSensorReg8_8(0xFF, 0x01);
    uint8_t val;
    ESP_LOGI(TAG, "--- SENSOR BANK ---");
    rdSensorReg8_8(0x0A, &val); ESP_LOGI(TAG, "PID [0x0A]: 0x%02X", val);
    rdSensorReg8_8(0x0B, &val); ESP_LOGI(TAG, "VER: 0x%02X", val);
    rdSensorReg8_8(0x11, &val); ESP_LOGI(TAG, "CLKRC [0x11]: 0x%02X", val);
    rdSensorReg8_8(0x12, &val); ESP_LOGI(TAG, "COM7 [0x12]: 0x%02X", val);
    rdSensorReg8_8(0x03, &val); ESP_LOGI(TAG, "COM1 [0x03]: 0x%02X", val);
    rdSensorReg8_8(0x04, &val); ESP_LOGI(TAG, "COM1 [0x04]: 0x%02X", val);
    rdSensorReg8_8(0x17, &val); ESP_LOGI(TAG, "HSTART [0x17]: 0x%02X", val);
    rdSensorReg8_8(0x18, &val); ESP_LOGI(TAG, "HSIZE [0x18]: 0x%02X", val);
    rdSensorReg8_8(0x19, &val); ESP_LOGI(TAG, "VSTART [0x19]: 0x%02X", val);
    rdSensorReg8_8(0x1A, &val); ESP_LOGI(TAG, "VSIZE [0x1A]: 0x%02X", val);

    // --- DSP BANK (0x00) ---
    wrSensorReg8_8(0xFF, 0x00);
    ESP_LOGI(TAG, "--- DSP BANK ---");
    rdSensorReg8_8(0xDA, &val); ESP_LOGI(TAG, "IMAGE_MODE: 0x%02X", val);
    rdSensorReg8_8(0xD3, &val); ESP_LOGI(TAG, "R_DVP_SP: 0x%02X", val);
    rdSensorReg8_8(0xC2, &val); ESP_LOGI(TAG, "CTRL2 [0xC2]: 0x%02X", val);
    rdSensorReg8_8(0x05, &val); ESP_LOGI(TAG, "R_BYPASS [0x05]: 0x%02X", val);
    rdSensorReg8_8(0x5A, &val); ESP_LOGI(TAG, "Y_SIZE [0x5A]: 0x%02X", val);
    rdSensorReg8_8(0x5B, &val); ESP_LOGI(TAG, "X_SIZE: 0x%02X", val);

    ESP_LOGI(TAG, "--- End of Dump ---");
}

// /* Initialize the OV2640 camera sensor. */
// void ov2640Init(void) {
//     // 1. Tell the ArduChip to clear its FIFO buffer in preparation for a new format
//     write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);

//     // 2. Now, send the full configuration sequence to the OV2640 sensor
//     // This list should be the complete YUV list from my first response,
//     // which includes the sensor software reset.
//     wrSensorRegs8_8(ov2640_yuv_qvga_config_regs);
//     vTaskDelay(pdMS_TO_TICKS(10));

//     ESP_LOGI("OV2640", "Initialization complete. Dumping final state.");
//     ov2640_dump_state();
// }

/* Prepares ESP for communication and initiates image capture. */
// This single function performs the entire initialization sequence correctly.
void arducam_yuv_init(void) {
    // 1. Initialize hardware peripherals (SPI and I2C)
    spi_master_init();
    i2c_master_init();
    uart_init();

    // 2. Test SPI communication with the ArduChip controller
    write_reg(ARDUCHIP_TEST1, 0x55);
    uint8_t test_val = read_reg(ARDUCHIP_TEST1);
    if (test_val!= 0x55) {
        ESP_LOGE("ARDUCAM", "SPI interface test failed! Halting.");
        while(1);
    }
    ESP_LOGI("ARDUCAM", "SPI interface OK.");

    // 3. Reset the ArduChip (CPLD) using the CTRL register (0x07).
    // This is the critical step for reliably changing image formats.
    write_reg(ARDUCHIP_GPIO_CTRL_REG, GPIO_CPLD_RESET_MASK);
    vTaskDelay(pdMS_TO_TICKS(100));
    write_reg(ARDUCHIP_GPIO_CTRL_REG, 0x00);
    vTaskDelay(pdMS_TO_TICKS(100));
    ESP_LOGI("ARDUCAM", "ArduChip reset complete.");

    // 4. Power up the OV2640 sensor using the GPIO WRITE register (0x06)
    ESP_LOGI("ARDUCAM", "Powering up sensor...");
    // Enable sensor LDO (power enable)
    write_reg(ARDUCHIP_GPIO_WRITE_REG, read_reg(ARDUCHIP_GPIO_WRITE_REG) | GPIO_PWREN_MASK);
    vTaskDelay(pdMS_TO_TICKS(10));
    // Ensure sensor is NOT in power-down mode (PWDN is active-high, so clear the bit)
    write_reg(ARDUCHIP_GPIO_WRITE_REG, read_reg(ARDUCHIP_GPIO_WRITE_REG) & ~GPIO_PWDN_MASK);
    vTaskDelay(pdMS_TO_TICKS(10));
    // Toggle sensor reset (RESETB is active-low)
    write_reg(ARDUCHIP_GPIO_WRITE_REG, read_reg(ARDUCHIP_GPIO_WRITE_REG) & ~GPIO_RESET_MASK);
    vTaskDelay(pdMS_TO_TICKS(10));
    write_reg(ARDUCHIP_GPIO_WRITE_REG, read_reg(ARDUCHIP_GPIO_WRITE_REG) | GPIO_RESET_MASK);
    vTaskDelay(pdMS_TO_TICKS(20));
    ESP_LOGI("ARDUCAM", "Sensor power up sequence complete.");

    // 5. Probe for the OV2640 sensor using the direct I2C bus
    wrSensorReg8_8(0xFF, 0x01); // Select SENSOR bank
    uint8_t idh = 0, idl = 0;
    rdSensorReg8_8(0x0A, &idh);
    rdSensorReg8_8(0x0B, &idl);
    if (idh == 0x26 && (idl == 0x41 || idl == 0x42)) {
        ESP_LOGI("OV2640", "OV2640 detected. ID=%02X%02X", idh, idl);
    } else {
        ESP_LOGE("OV2640", "OV2640 not found. ID=%02X%02X", idh, idl);
        while(1); // Halt on critical failure
    }

    // 6. Apply the full YUV configuration to the sensor via the direct I2C bus
    wrSensorRegs8_8(ov2640_yuv_qvga_init_regs);
    vTaskDelay(pdMS_TO_TICKS(10));

    ESP_LOGI("OV2640", "YUV initialization complete.");
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

// This function will help us isolate the problem.
// void arducam_minimal_test(void) {
//     // 1. Initialize hardware peripherals
//     spi_master_init();
//     uart_init();

//     // 2. Test SPI communication with the ArduChip controller
//     write_reg(ARDUCHIP_TEST1, 0x55);
//     uint8_t test_val = read_reg(ARDUCHIP_TEST1);
//     if (test_val!= 0x55) {
//         ESP_LOGE("ARDUCAM", "SPI interface test failed! Halting.");
//         while(1);
//     }
//     ESP_LOGI("ARDUCAM", "SPI interface OK.");

//     // 3. Power up the OV260 sensor using the GPIO WRITE register (0x06)
//     // We are using longer delays here for stability testing.
//     ESP_LOGI("ARDUCAM", "Powering up sensor...");
//     uint8_t gpio_val = 0;

//     // Enable sensor LDO (power enable)
//     gpio_val = read_reg(ARDUCHIP_GPIO_WRITE_REG);
//     write_reg(ARDUCHIP_GPIO_WRITE_REG, gpio_val | GPIO_PWREN_MASK);
//     vTaskDelay(pdMS_TO_TICKS(100));

//     // Ensure sensor is NOT in power-down mode (PWDN is active-high, so clear the bit)
//     gpio_val = read_reg(ARDUCHIP_GPIO_WRITE_REG);
//     write_reg(ARDUCHIP_GPIO_WRITE_REG, gpio_val & ~GPIO_PWDN_MASK);
//     vTaskDelay(pdMS_TO_TICKS(100));
//     ESP_LOGI("ARDUCAM", "Sensor power up sequence complete.");

//     // 4. Attempt to read the sensor ID.
//     // This is the most critical test.
//     ESP_LOGI("OV2640", "Attempting to read sensor ID...");
//     wrSensorReg8_8(0xFF, 0x01); // Select SENSOR bank
//     uint8_t idh = 0, idl = 0;
//     rdSensorReg8_8(0x0A, &idh);
//     rdSensorReg8_8(0x0B, &idl);

//     if (idh == 0x26 && (idl == 0x41 || idl == 0x42)) {
//         ESP_LOGI("OV2640", "SUCCESS! OV2640 detected. ID=%02X%02X", idh, idl);
//     } else {
//         ESP_LOGE("OV2640", "FAILURE. OV2640 not found. ID=%02X%02X", idh, idl);
//     }

//     // We will stop here. The program will do nothing further.
//     while(1) {
//         vTaskDelay(pdMS_TO_TICKS(1000));
//     }
// }

struct camera_operate arducam = {
   .slave_address = 0x30,
   .init          = arducam_yuv_init, // Point the single init function here
   .setJpegSize   = OV2640_set_JPEG_size,
};
