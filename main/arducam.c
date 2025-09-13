
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
#include "freertos/semphr.h"

static SemaphoreHandle_t spi_mutex;


#include "arducam.h"
#include "ov2640.h"
#include "wifi_cam.h"

void set_bit(unsigned char addr, unsigned char bit);
void clear_bit(unsigned char addr, unsigned char bit);

i2c_master_bus_handle_t bus_handle;
i2c_master_dev_handle_t camera_dev_handle;

spi_device_handle_t spi_device_handle;

volatile uint8_t cameraCommand = 0;
static QueueHandle_t uart_queue;
static const char* TAG = "UART_HANDLER";
static uint8_t *dummy_tx;

static inline void spi_lock(void)   { if (spi_mutex) xSemaphoreTake(spi_mutex, portMAX_DELAY); }
static inline void spi_unlock(void) { if (spi_mutex) xSemaphoreGive(spi_mutex); }

void i2c_master_init(){ // peripherals/i2c/i2c_basic
    // Initialize i2C bus for the camera device
    i2c_master_bus_config_t bus_config = {};
        bus_config.i2c_port = I2C_MASTER_NUM;
        bus_config.sda_io_num = PIN_SDA;
        bus_config.scl_io_num = PIN_SCL;
        bus_config.clk_source = I2C_CLK_SRC_DEFAULT;
        bus_config.glitch_ignore_cnt = 7;
        bus_config.flags.enable_internal_pullup = true;
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_config, &bus_handle));

    i2c_device_config_t dev_config = {};
    dev_config.dev_addr_length = I2C_ADDR_BIT_LEN_7;
    dev_config.scl_speed_hz = I2C_MASTER_FREQ_HZ;

    dev_config.device_address = arducam.slave_address;
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus_handle, &dev_config, &camera_dev_handle));
}

void spi_master_init(void) {
    spi_bus_config_t buscfg = {
        .mosi_io_num = PIN_MOSI,
        .miso_io_num = PIN_MISO,
        .sclk_io_num = PIN_SCK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = SPI_CHUNK,   // OK; we’ll chunk to this
    };
    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &buscfg, SPI_DMA_CH_AUTO));

    spi_device_interface_config_t devcfg = {
        .command_bits = 0,
        .address_bits = 0,
        .dummy_bits = 0,
        .clock_speed_hz = 8 * 1000 * 1000,
        .mode = 0,
        .spics_io_num = PIN_CS,
        .queue_size = 7,
        .flags = 0,                // <-- back to full-duplex
    };
    ESP_ERROR_CHECK(spi_bus_add_device(SPI2_HOST, &devcfg, &spi_device_handle));
}

void uart_init(void) {
    uart_config_t uart_config = {
        .baud_rate = BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    // Install driver and ask it to create an event queue of 20 items
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, 256 * 2, 0, 20, &uart_queue, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM, UART_TX_PIN, UART_RX_PIN,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

void uart_event_task(void *pvParameters) { // uart/events/example
    uart_event_t event;
    uint8_t* dtmp = (uint8_t*) malloc(RD_BUF_SIZE);
    
    for(;;) {
        // Wait for the next event in the queue
        if(xQueueReceive(uart_queue, (void * )&event, (TickType_t)portMAX_DELAY)) {
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

void esp32c3_SystemInit(void) {
    i2c_master_init();
    spi_master_init();
    uart_init();
    if (!spi_mutex) spi_mutex = xSemaphoreCreateMutex();
    arducam_power_up_sensor();
}

int rdSensorReg8_8(uint8_t regID, uint8_t* regDat) {
    esp_err_t err = i2c_master_transmit_receive(
        camera_dev_handle, &regID, 1, regDat, 1, pdMS_TO_TICKS(I2C_TIMEOUT_MS)
    );
    if (err != ESP_OK) {
        ESP_LOGE("SCCB", "Read reg 0x%02X failed: %s", regID, esp_err_to_name(err));
    }
    return err;
}

int wrSensorReg8_8(uint8_t regID, uint8_t regDat) {
    uint8_t buf[2] = {regID, regDat};
    esp_err_t err = i2c_master_transmit(camera_dev_handle, buf, 2, pdMS_TO_TICKS(I2C_TIMEOUT_MS));
    if (err != ESP_OK) {
        ESP_LOGE("SCCB", "Write reg 0x%02X=0x%02X failed: %s", regID, regDat, esp_err_to_name(err));
    }
    return err;
}

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

uint8_t read_reg(uint8_t address) {
    spi_lock();
    uint8_t tx[2] = { (uint8_t)(address & 0x7F), 0x00 };
    uint8_t rx[2] = {0};
    spi_transaction_t t = {0};
    t.length    = 16;
    t.rxlength  = 16;
    t.tx_buffer = tx;
    t.rx_buffer = rx;
    esp_err_t err = spi_device_polling_transmit(spi_device_handle, &t);
    spi_unlock();
    ESP_ERROR_CHECK(err);
    return rx[1];
}

int wrSensorRegs8_8(const struct sensor_reg reglist[]) {
    int err = 0;
    const struct sensor_reg *next = reglist;

    // Check the next entry *before* processing it.
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

unsigned char read_fifo(void)
{
	unsigned char data;
	data = read_reg(SINGLE_FIFO_READ);
	return data;
}

static inline void ensure_dummy(void) {
    if (!dummy_tx) {
        dummy_tx = heap_caps_malloc(SPI_CHUNK, MALLOC_CAP_DMA);
        assert(dummy_tx);
        memset(dummy_tx, 0, SPI_CHUNK);
    }
}

void set_fifo_burst(uint8_t *buffer, uint32_t length) {
    ensure_dummy();

    spi_lock();  // block other tasks using our device
    // Keep the hardware bus exclusively during the whole burst
    ESP_ERROR_CHECK(spi_device_acquire_bus(spi_device_handle, portMAX_DELAY));

    // 1) Send BURST command, keep CS low
    uint8_t cmd = BURST_FIFO_READ;
    spi_transaction_t t0 = (spi_transaction_t){0};
    t0.length    = 8;
    t0.tx_buffer = &cmd;
    t0.flags     = SPI_TRANS_CS_KEEP_ACTIVE;
    ESP_ERROR_CHECK(spi_device_polling_transmit(spi_device_handle, &t0));

    // 2) Read all data in ≤4092-byte chunks; keep CS low until last
    while (length > 0) {
        uint32_t n = (length > SPI_CHUNK) ? SPI_CHUNK : length;

        spi_transaction_t t = (spi_transaction_t){0};
        t.length    = n * 8;       // clock n bytes (full-duplex)
        t.rxlength  = n * 8;
        t.tx_buffer = dummy_tx;    // real buffer → avoids txdata path
        t.rx_buffer = buffer;
        t.flags     = (n == length) ? 0 : SPI_TRANS_CS_KEEP_ACTIVE;

        ESP_ERROR_CHECK(spi_device_polling_transmit(spi_device_handle, &t));

        buffer += n;
        length -= n;
    }

    spi_device_release_bus(spi_device_handle);
    spi_unlock();
}

void flush_fifo(void)
{
	write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
}

void start_capture(void)
{
	write_reg(ARDUCHIP_FIFO, FIFO_START_MASK);
}

void clear_fifo_flag(void )
{
	write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
}

unsigned int read_fifo_length()
{
    unsigned int len1,len2,len3,len=0;
    len1 = read_reg(FIFO_SIZE1);
    len2 = read_reg(FIFO_SIZE2);
    len3 = read_reg(FIFO_SIZE3) & 0x7f;
    len = ((len3 << 16) | (len2 << 8) | len1) & 0x07fffff;
	return len;	
}

//Set corresponding bit  
void set_bit(unsigned char addr, unsigned char bit)
{
	unsigned char temp;
	temp = read_reg(addr);
	write_reg(addr, temp | bit);
}
//Clear corresponding bit 
void clear_bit(unsigned char addr, unsigned char bit)
{
	unsigned char temp;
	temp = read_reg(addr);
	write_reg(addr, temp & (~bit));
}

//Get corresponding bit status
unsigned char get_bit(unsigned char addr, unsigned char bit)
{
  unsigned char temp;
  temp = read_reg(addr);
  temp = temp & bit;
  return temp;
}

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

static inline void fifo_reset_all(void) {
    // Clear + reset read/write pointers
    write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK | FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
}

void singleCapture(void) {
    // 1) Prepare FIFO and start capture
    fifo_reset_all();
    start_capture();

    // 2) Wait for capture done
    while (!get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
        vTaskDelay(pdMS_TO_TICKS(1));
    }

    // 3) Read JPEG length and validate
    uint32_t length = read_fifo_length();
    if (length == 0 || length > WIFI_CAM_MAX_JPEG) {
        ESP_LOGE("cam", "Bad FIFO length: %lu (max %u)", (unsigned long)length, WIFI_CAM_MAX_JPEG);
        return;
    }

    // 4) Allocate exactly what we need (DMA-capable is fine)
    uint8_t *image_buffer = heap_caps_malloc(length, MALLOC_CAP_DMA);
    if (!image_buffer) {
        ESP_LOGE("cam", "No memory for %lu bytes", (unsigned long)length);
        return;
    }

    // 5) Burst-read the JPEG into RAM
    set_fifo_burst(image_buffer, length);

    // (Optional) quick sanity check of JPEG markers
    bool soi = (length >= 2 && image_buffer[0] == 0xFF && image_buffer[1] == 0xD8);
    bool eoi = (length >= 2 && image_buffer[length-2] == 0xFF && image_buffer[length-1] == 0xD9);
    ESP_LOGI("cam", "JPEG len=%lu SOI=%d EOI=%d", (unsigned long)length, soi, eoi);

    // 6) Publish to the web handler and free
    wifi_cam_publish(image_buffer, length);
    free(image_buffer);
}

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

struct camera_operate arducam = {
    .slave_address = 0x30,
    .systemInit    = esp32c3_SystemInit, // Point to your new ESP32 init function
    .busDetect     = spiBusDetect,
    .cameraProbe   = ov2640Probe,
    .cameraInit    = ov2640Init,
    .setJpegSize   = OV2640_set_JPEG_size,
};
