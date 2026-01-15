// arducam.c
// Bradley Manzo 2026
#include "includes/arducam.h"
#include "includes/globals.h"
#include "includes/i2c.h"
#include "includes/ov2640.h"
#include "includes/spi.h"

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
    // Software Reset
    i2c_write_reg(0xFF, 0x01);
    i2c_write_reg(0x12, 0x80);
    vTaskDelay(pdMS_TO_TICKS(100));

    // Load Base Configs
    i2c_write_regs(OV2640_JPEG_INIT); 
    i2c_write_regs(OV2640_YUV422);

    // Load Resolution
    i2c_write_regs(OV2640_320x240_JPEG);

    // Force DSP to Output RAW YUV422 (Y first)
    i2c_write_reg(0xFF, 0x00); // Select Bank 0 (DSP)

    uint8_t reg_DA;
    i2c_read_reg(0xDA, &reg_DA);

    // No JPEG (Bit 4=0) AND No Byte Swap (Bit 0=0)
    reg_DA &= 0xEE; // Mask: 1110 1110 (Clears Bit 4 and Bit 0)

    i2c_write_reg(0xDA, reg_DA);

    // Short settlement delay
    vTaskDelay(pdMS_TO_TICKS(50));
    
    ESP_LOGI("OV2640", "Init complete. DSP Ctrl (0xDA): 0x%02X", reg_DA);
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
    .systemInit    = esp32c3_SystemInit,
    .busDetect     = spi_bus_detect,
    .cameraProbe   = ov2640Probe,
    .cameraInit    = ov2640Init,
    .setJpegSize   = OV2640_set_JPEG_size,
};

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


/* Reset the Arducam capture FIFO */
void arducam_reset_fifo(void) {
    spi_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK); 
    esp_rom_delay_us(10);
    spi_write_reg(ARDUCHIP_FIFO, 0x00); 

    // Completely reset Arducam FIFO
    spi_write_reg(ARDUCHIP_FIFO, FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
    spi_write_reg(ARDUCHIP_TRIG, CAP_DONE_MASK); 
    if (spi_read_fifo_len() > 0) {
        ESP_LOGW("cam", "FIFO stubborn! Force clearing again.");
        spi_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
        spi_write_reg(ARDUCHIP_FIFO, 0x00);
        spi_write_reg(ARDUCHIP_FIFO, FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
    }
    spi_write_reg(ARDUCHIP_TRIG, CAP_DONE_MASK); 

    // Ensure that FIFO is empty before capture
    if (spi_read_fifo_len() > 0) {
        ESP_LOGW("cam", "FIFO not empty after reset! Force clearing again.");
        spi_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
        spi_write_reg(ARDUCHIP_FIFO, 0x00);
    }
}

/* Set the Arducam to capture mode. 0x01 */
void arducam_set_capture(void) {
   spi_write_reg(ARDUCHIP_MODE, 0x01);
}

/* Start the image capture. */
void arducam_start_capture(void) {
	spi_write_reg(ARDUCHIP_FIFO, FIFO_START_MASK);
}

/* Stops Arducam Capture and prints recieved image length from FIFO */
void arducam_stop_capture(void) {
    spi_write_reg(ARDUCHIP_FIFO, 0x00); // Kill Write Enable
    spi_write_reg(ARDUCHIP_FIFO, FIFO_RDPTR_RST_MASK); // Reset Read Pointer to 0

    // Read FIFO length
    uint32_t len = spi_read_fifo_len();
    ESP_LOGI("cam", "Capture Done. FIFO Len: %u", (unsigned)len);
}

/* Convert Luma (Y) value to 1-bit using threshold. */
static inline uint8_t luma_to_bit(uint8_t y, uint8_t threshold) {
    return (y > threshold) ? 1 : 0;
}

/* Reads Arducam FIFO and packs Luma (Y) values, 1 bit per pixel onto a byte. */
esp_err_t arducam_read_and_pack_stream(uint8_t *out, size_t out_cap, uint16_t w, uint16_t h, uint8_t* adaptive_th, uint8_t capture_num)
{
    const size_t npix    = (size_t)w * h;
    const size_t out_len = (npix + 7) / 8;
    const size_t raw_len = npix * 2;
    if (out_cap < out_len) return ESP_ERR_NO_MEM;

    // 1. LOCK THE BUS
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

    // 3. BURN DUMMY BYTE
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
    bool keep = false; // Skip U, Keep Y (assuming correct phase from previous fix)

    // MODE A: PACKING (Using existing threshold)
    if (capture_num < RECALIBRATE_INTERVAL) {
        while (remaining) {
            size_t n = remaining > RJPEG_PULL_CHUNK ? RJPEG_PULL_CHUNK : remaining;
            bool keep_cs = (remaining > n); 
            
            e = spi_read_chunk(tmp, n, keep_cs);
            if (e != ESP_OK) { spi_device_release_bus(spi_device_handle); return e; }

            for (size_t i = 0; i < n; i++) {
                if (keep) {
                    // Corrected syntax: Pass dereferenced threshold, shift RESULT
                    acc |= (luma_to_bit(tmp[i], *adaptive_th) << bitpos);
                    
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
        // Cleanup trailing bits
        if (bitpos != 0 && out_i < out_len) out[out_i++] = acc;
    } 
    
    // MODE B: CALIBRATION (Calculating average)
    else {
        uint64_t sum_luma = 0;
        size_t count_luma = 0;

        while (remaining) {
            size_t n = remaining > RJPEG_PULL_CHUNK ? RJPEG_PULL_CHUNK : remaining;
            bool keep_cs = (remaining > n);

            e = spi_read_chunk(tmp, n, keep_cs);
            if (e != ESP_OK) { spi_device_release_bus(spi_device_handle); return e; }

            for (size_t i = 0; i < n; i++) {
                if (keep) {
                    sum_luma += tmp[i]; // Accumulate brightness
                    count_luma++;
                }
                keep = !keep;
            }
            remaining -= n;
        }

        // Compute Average
        if (count_luma > 0) {
            uint8_t avg = (uint8_t)(sum_luma / count_luma);
            uint8_t bias = 0; // Bias to favor darker pixels
            *adaptive_th = avg + bias;
            if (avg > 230) *adaptive_th = 255;

            // Min fallback 10 ensures next frame goes to 'Packing' mode.
            if (*adaptive_th < 10) *adaptive_th = 10;
            
            printf("Calibration Done. Avg Luma: %u\n", (unsigned)*adaptive_th);
        }
        
        // Fill output with 0 or a dummy pattern so we don't send garbage
        memset(out, 0, out_len); 
    }

    spi_device_release_bus(spi_device_handle);
    return ESP_OK;
}