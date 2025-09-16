#include <stdio.h>
#include <string.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/i2c_master.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "driver/uart.h"

#include "arducam.h"
#include "wifi_cam.h"

static void camera_task(void *arg) {
    for (;;) {
        singleCapture();                 // this calls wifi_cam_publish(...)
        vTaskDelay(pdMS_TO_TICKS(800));  // let Wi-Fi/HTTP breathe
    }
}

void app_main(void) {
    // 1) Initialize the entire Arducam system.
    // This single function handles SPI bus checks, CPLD reset, sensor power-up,
    // camera probing, and the final YUV configuration.
    arducam_minimal_test();
    arducam.init();

    // 2) (Optional) UART RX task — keep priority modest (e.g., 5–7)
    xTaskCreate(uart_event_task, "uart_event_task", 4096, NULL, 5, NULL);

    // 3) Set the desired image size.
    // Note: The arducam.init() function already configures the camera for QVGA (320x240).
    // This line is only necessary if you intend to immediately switch to a different JPEG size.
    // For YUV processing, this line can be removed.
    arducam.setJpegSize(res_320x240);

    // 4) Bring up SoftAP + HTTP ONCE
    ESP_ERROR_CHECK(wifi_cam_init("esp-cam", "12345678"));
    vTaskDelay(pdMS_TO_TICKS(800));  // Grace period so AP/HTTP are ready

    // 5) Start the main capture loop task
    xTaskCreate(camera_task, "camera_task", 6144, NULL, 6, NULL);
}