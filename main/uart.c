#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "driver/uart.h"

#include "includes/arducam.h"
#include "includes/uart.h"

/* -------------------------------------- UART -------------------------------------- */
/* Initialize UART for streaming images to interfaces. */

void uart_init(void) {
    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    // We won't use a buffer for sending data.
    uart_driver_install(UART_NUM_1, 4096, 0, 0, NULL, 0);
    uart_param_config(UART_NUM_1, &uart_config);
    uart_set_pin(UART_NUM_1, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
}

esp_err_t uart_read_exact(uart_port_t uart, uint8_t *dst, size_t n)
{
    size_t got = 0;
    while (got < n) {
        int r = uart_read_bytes(uart, dst + got, n - got, pdMS_TO_TICKS(1000));
        if (r < 0) return ESP_FAIL;
        if (r == 0) continue;
        got += (size_t)r;
    }
    return ESP_OK;
}

esp_err_t uart_write_all(uart_port_t uart, const uint8_t *buf, size_t len)
{
    size_t off = 0;
    while (off < len) {
        int w = uart_write_bytes(uart, (const char*)(buf + off), len - off);
        if (w < 0) return ESP_FAIL;
        off += (size_t)w;
    }
    return ESP_OK;
}