#ifndef SPI_H
#define SPI_H

#include <stddef.h>
#include <stdbool.h>

#include "esp_err.h"
#include "driver/spi_master.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------- SPI -------------------------------------- */
/* Initialize SPI master */
void spi_master_init(void);

/* Write to device register using SPI. */
void spi_write_reg(uint8_t address, uint8_t value);

/* Read from device register using SPI. */
uint8_t spi_read_reg(uint8_t address);

/* Set bit at address using SPI. */
void spi_set_bit(unsigned char addr, unsigned char bit);
/* Clear bit at address using SPI. */
void spi_clear_bit(unsigned char addr, unsigned char bit);

/* Get bit at address using SPI. */
unsigned char spi_get_bit(unsigned char addr, unsigned char bit);

/* Start the image capture. */
void start_capture(void);

/* Read the length of data in the SPI FIFO buffer. */
unsigned int spi_read_fifo_len(void);

/* Reset the SPI FIFO buffer. */
void spi_reset_fifo(void);

/* Reads exactly n bytes from SPI FIFO */
esp_err_t spi_read_chunk(uint8_t *dst, size_t n, bool keep_cs);

#ifdef __cplusplus
}
#endif

#endif // SPI_H