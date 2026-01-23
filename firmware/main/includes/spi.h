// spi.h
// Bradley Manzo 2026
#ifndef SPI_H
#define SPI_H

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

/* Set bit at address using SPI. */
void spi_set_bit(unsigned char addr, unsigned char bit);
/* Clear bit at address using SPI. */
void spi_clear_bit(unsigned char addr, unsigned char bit);

/* Get bit at address using SPI. */
unsigned char spi_get_bit(unsigned char addr, unsigned char bit);

/* Read the length of data in the SPI FIFO buffer. */
unsigned int spi_read_fifo_len(void);

/* Detect the SPI bus operational state. */
uint8_t spi_bus_detect(void);

/* Reads exactly n bytes from SPI FIFO */
esp_err_t spi_read_chunk(uint8_t *dst, size_t n, bool keep_cs);

#ifdef __cplusplus
}
#endif

#endif // SPI_H