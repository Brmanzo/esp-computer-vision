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
    // 1) Board bring-up
    arducam.systemInit();

    // 2) (Optional) UART RX task — keep priority modest (e.g., 5–7)
    xTaskCreate(uart_event_task, "uart_event_task", 4096, NULL, 5, NULL);

    // 3) Probe/init camera
    if (arducam.busDetect() != 0) { ESP_LOGE("main","SPI bus test failed."); return; }
    if (arducam.cameraProbe() != 0){ ESP_LOGE("main","Camera sensor probe failed."); return; }
    arducam.cameraInit();
    arducam.setJpegSize(res_160x120);

    // 4) Bring up SoftAP + HTTP ONCE
    ESP_ERROR_CHECK(wifi_cam_init("esp-cam", "12345678"));
    vTaskDelay(pdMS_TO_TICKS(800));  // grace period so AP/HTTP are ready

    // 5) Start capture loop task (priority 6 is usually safe)
    xTaskCreate(camera_task, "camera_task", 6144, NULL, 6, NULL);
}