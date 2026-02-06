// uart.h
// Bradley Manzo 2026
#ifndef UART_H
#define UART_H

#include "driver/uart.h"
#include "includes/gpio.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------- UART -------------------------------------- */

/* Initialize UART for streaming camera image to Icebreaker FPGA */
void uart_init(void);

/* Write exactly len bytes to Icebreaker FPGA */
esp_err_t uart_write_all(uart_port_t uart, const uint8_t *buf, size_t len);

/* Wait for wakeup byte from FPGA */
bool uart_wait_for_wakeup_byte(uart_port_t uart, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif // UART_H