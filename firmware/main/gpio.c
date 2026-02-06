// gpio.c
// Bradley Manzo 2026
#include "includes/arducam.h"
#include "includes/wifi_cam.h"
#include "includes/capture.h"
#include "includes/uart.h"
#include "driver/gpio.h"

void gpio_init(void) {
    gpio_config_t io = {
        .pin_bit_mask = 1ULL << GPIO_RESET_FPGA,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };

    gpio_config(&io);

    // HOLD FPGA in reset while ESP brings up UART
    gpio_set_level(GPIO_RESET_FPGA, 0);
}

void fpga_reset(void) {
    uart_flush_input(UART_NUM_1);
    gpio_set_level(GPIO_RESET_FPGA, 0);
    vTaskDelay(pdMS_TO_TICKS(1000));
    gpio_set_level(GPIO_RESET_FPGA, 1);
}