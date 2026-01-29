// uart.c
// Bradley Manzo 2026
#include "driver/uart.h"

#include "includes/arducam.h"
#include "includes/uart.h"
#include "driver/gpio.h"

#define UART_RTS_PIN GPIO_NUM_1

/* Initialize UART for streaming images to interfaces. */
void uart_init(void) {
    const uart_config_t uart_config = {
        .baud_rate = 312500,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_CTS,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_DEFAULT,
    };
    // We won't use a buffer for sending data.
    uart_driver_install(UART_NUM_1, 4096, 0, 0, NULL, 0);
    uart_param_config(UART_NUM_1, &uart_config);
    uart_set_pin(UART_NUM_1, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_RTS_PIN);
}

/* Write all data from buffer to UART in order. */
esp_err_t uart_write_all(uart_port_t uart, const uint8_t *buf, size_t len)
{
    size_t idx = 0;
    while (idx < len) {
        int w = uart_write_bytes(uart, (const char*)&buf[idx], len - idx);
        if (w < 0) return ESP_FAIL;
        idx += (size_t)w;
    }
    return ESP_OK;
}