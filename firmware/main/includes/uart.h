// uart.h
// Bradley Manzo 2026
#ifndef UART_H
#define UART_H

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

#ifdef __cplusplus
}
#endif

#endif // UART_H