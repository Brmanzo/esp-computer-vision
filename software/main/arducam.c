#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "includes/arducam.h"
#include "includes/ov2640.h"
#include "includes/spi.h"
#include "includes/i2c.h"
#include "includes/uart.h"


/* -------------------------------------- Device Init -------------------------------------- */
/* Initialization sequence for the ArduCAM-M-2MP device. */
void arducam_power_up_sensor(void) {
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

/* Initialize the OV2640 camera sensor for RAW YUV422 320x240. */
void ov2640Init(void){
    // 1. Software Reset
    i2c_write_reg(0xFF, 0x01);
    i2c_write_reg(0x12, 0x80);
    vTaskDelay(pdMS_TO_TICKS(100));

    // 2. Load Base Configs
    // We use the JPEG init arrays because they contain the necessary 
    // electrical setup (clocks, gain, etc), but we will strip the JPEG mode later.
    i2c_write_regs(OV2640_JPEG_INIT); 
    i2c_write_regs(OV2640_YUV422);

    // 3. Load Resolution (CRITICAL!)
    // You MUST load this to set the PCLK and Window Size correctly.
    // NOTE: This array inevitably re-enables JPEG (Bit 4 of 0xDA), 
    // so we must patch 0xDA *after* this step.
    i2c_write_regs(OV2640_320x240_JPEG);

    // 4. Force DSP to Output RAW YUV422 (Y first)
    i2c_write_reg(0xFF, 0x00); // Select Bank 0 (DSP)

    uint8_t reg_DA;
    i2c_read_reg(0xDA, &reg_DA);

    // STRICTLY enforce: No JPEG (Bit 4=0) AND No Byte Swap (Bit 0=0)
    reg_DA &= 0xEE; // Mask: 1110 1110 (Clears Bit 4 and Bit 0)

    i2c_write_reg(0xDA, reg_DA);

    // 5. Short settlement delay
    vTaskDelay(pdMS_TO_TICKS(50));
    
    ESP_LOGI("OV2640", "Init complete. DSP Ctrl (0xDA): 0x%02X", reg_DA);
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
    .systemInit    = esp32c3_SystemInit,
    .busDetect     = spiBusDetect,
    .cameraProbe   = ov2640Probe,
    .cameraInit    = ov2640Init,
    .setJpegSize   = OV2640_set_JPEG_size,
};