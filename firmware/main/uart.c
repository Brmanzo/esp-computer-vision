// uart.c
// Bradley Manzo 2026
#include <stdbool.h>
#include "driver/uart.h"

#include "includes/arducam.h"
#include "includes/uart.h"
#include "driver/gpio.h"

#define UART_RTS_PIN GPIO_NUM_1
#define FPGA_WAKEUP_BYTE 0x99

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

bool uart_wait_for_wakeup_byte(uart_port_t uart, uint32_t timeout_ms) {
    const TickType_t start = xTaskGetTickCount();
    const TickType_t timeout_ticks = pdMS_TO_TICKS(timeout_ms);
    uint8_t b = 0;

    for (;;) {
        int n = uart_read_bytes(uart, &b, 1, pdMS_TO_TICKS(20));
        if (n == 1) {
            if (b == FPGA_WAKEUP_BYTE) {
                ESP_LOGI("uart", "Received wakeup byte from FPGA");
                return true;
            } else {
                ESP_LOGW("uart", "Received unexpected byte 0x%02X while waiting for wakeup byte", b);
            }
        }
        if (timeout_ms != 0 && (xTaskGetTickCount() - start) > timeout_ticks) {
            ESP_LOGE("uart", "Timed out waiting for FPGA wakeup byte (0x%02X)", FPGA_WAKEUP_BYTE);
            return false;
        }
    }
}