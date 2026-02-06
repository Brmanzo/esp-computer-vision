// i2c.c
// Bradley Manzo 2026
#include "driver/i2c_master.h"

#include "includes/i2c.h"
#include "includes/globals.h"
#include "includes/gpio.h"

/* -------------------------------------- I2C -------------------------------------- */
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
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_config, &i2c_bus_handle));

    i2c_device_config_t dev_config = {};
    dev_config.dev_addr_length = I2C_ADDR_BIT_LEN_7;
    dev_config.scl_speed_hz = I2C_MASTER_FREQ_HZ;

    dev_config.device_address = arducam.slave_address;
    ESP_ERROR_CHECK(i2c_master_bus_add_device(i2c_bus_handle, &dev_config, &camera_dev_handle));
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
int i2c_write_regs(const struct sensor_reg *regs)
{
    for (int i = 0; ; i++) {
        uint8_t reg = regs[i].reg;
        uint8_t val = regs[i].val;

        if (reg == 0xFF && val == 0xFF) {   // end marker
            break;
        }

        esp_err_t err = i2c_write_reg(reg, val);
        vTaskDelay(pdMS_TO_TICKS(1));
        if (err != ESP_OK) return err;

        // Optional: tiny delay helps some sensors/bridges
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    return ESP_OK;
}