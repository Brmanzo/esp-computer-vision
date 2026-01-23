// i2c.h
// Bradley Manzo 2026
#ifndef I2C_H
#define I2C_H

#include "arducam.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------- I2C -------------------------------------- */

/* Initialize i2c for configuring the ArduCAM-M-2MP device. */
void i2c_master_init();

/* Read a 8-bit register from the sensor. */
int i2c_read_reg(uint8_t regID, uint8_t* regDat);

/* Write an 8-bit value to an 8-bit register on the sensor. */
int i2c_write_reg(uint8_t regID, uint8_t regDat);

/* Write multiple 8-bit values to 8-bit registers on the sensor. */
int i2c_write_regs(const struct sensor_reg *regs);

#ifdef __cplusplus
}
#endif

#endif // I2C_H