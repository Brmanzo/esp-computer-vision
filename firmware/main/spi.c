// spi.c
// Bradley Manzo 2026
#include "driver/spi_master.h"

#include "includes/arducam.h"
#include "includes/spi.h"
#include "includes/globals.h"

/* -------------------------------------- SPI -------------------------------------- */
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

/* Write to device register using SPI. */
void spi_write_reg(uint8_t address, uint8_t value) {
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
uint8_t spi_read_reg(uint8_t address) {
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
void spi_set_bit(unsigned char addr, unsigned char bit)
{
	unsigned char tmp;
	tmp = spi_read_reg(addr);
	spi_write_reg(addr, tmp | bit);
}

/* Clear bit at address using SPI. */
void spi_clear_bit(unsigned char addr, unsigned char bit)
{
	unsigned char tmp;
	tmp = spi_read_reg(addr);
	spi_write_reg(addr, tmp & (~bit));
}

/* Get bit at address using SPI. */
unsigned char spi_get_bit(unsigned char addr, unsigned char bit)
{
  unsigned char tmp;
  tmp = spi_read_reg(addr);
  tmp = tmp & bit;
  return tmp;
}

/* Read the length of data in the SPI FIFO buffer. */
unsigned int spi_read_fifo_len()
{
    unsigned int len1,len2,len3,len=0;
    len1 = spi_read_reg(FIFO_SIZE1);
    len2 = spi_read_reg(FIFO_SIZE2);
    len3 = spi_read_reg(FIFO_SIZE3) & 0x7f;
    len = ((len3 << 16) | (len2 << 8) | len1) & 0x07fffff;
	return len;	
}

/* Reset the SPI FIFO buffer. */
void spi_reset_fifo(void) {
    // Clear + reset read/write pointers
    spi_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK | FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
}


/* Detect the SPI bus operational state. */
uint8_t spi_bus_detect(void){
    spi_write_reg(0x00, 0x55);
    if(spi_read_reg(0x00) == 0x55){
        printf("SPI bus normal");
        return 0;
    }else{
        printf("SPI bus error\r\n");
        return 1;
    }      
}

/* Reads chunk of data from SPI FIFO without yielding */
esp_err_t spi_read_chunk(uint8_t *dst, size_t n, bool keep_cs)
{
    size_t done = 0;
    while (done < n) {
        size_t to = n - done;
        if (to > RJPEG_PULL_CHUNK) to = RJPEG_PULL_CHUNK;

        spi_transaction_t t = {
            .length    = to * 8,
            .rxlength  = to * 8,
            .tx_buffer = dummy_tx,
            .rx_buffer = dst + done,
            .flags     = ((keep_cs || (done + to < n)) ? SPI_TRANS_CS_KEEP_ACTIVE : 0)
        };

        esp_err_t e = spi_device_polling_transmit(spi_device_handle, &t);
        if (e != ESP_OK) return e;

        done += to;
    }
    return ESP_OK;
}