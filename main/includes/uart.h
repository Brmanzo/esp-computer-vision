#ifndef UART_H
#define UART_H

#include <stdint.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "driver/uart.h"


#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------- UART -------------------------------------- */
#define TXD_PIN (GPIO_NUM_21)
#define RXD_PIN (GPIO_NUM_20)

/* Initialize UART for streaming camera image to Icebreaker FPGA */
void uart_init(void);

/* Write exactly len bytes to Icebreaker FPGA */
esp_err_t uart_write_all(uart_port_t uart, const uint8_t *buf, size_t len);

/* Read exactly n bytes from Icebreaker FPGA */
esp_err_t uart_read_exact(uart_port_t uart, uint8_t *dst, size_t n);


#ifdef __cplusplus
}
#endif

#endif // UART_H