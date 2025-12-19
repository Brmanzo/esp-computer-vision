#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#include "includes/arducam.h"
#include "includes/wifi_cam.h"
#include "includes/capture.h"
#include "includes/uart.h"

static void camera_task(void *arg) {
    for (;;) {
        singleCapture();
        vTaskDelay(pdMS_TO_TICKS(800));
    }
}

void app_main(void) {
    // Board bring-up
    arducam.systemInit();

    // Initialize UART to Icebreaker FPGA
    uart_init();

    // Probe and initialize ArduCAM + OV2640
    if (arducam.busDetect() != 0) { ESP_LOGE("main","SPI bus test failed."); return; }
    if (arducam.cameraProbe() != 0){ ESP_LOGE("main","Camera sensor probe failed."); return; }
    arducam.cameraInit();
    arducam.setJpegSize(res_160x120);

    // Bring up SoftAP + HTTP ONCE
    ESP_ERROR_CHECK(wifi_cam_init(DEFAULT_SSID, DEFAULT_PASS));
    vTaskDelay(pdMS_TO_TICKS(800));  // grace period so AP/HTTP are ready

    // Start capture loop task (priority 6 is usually safe)
    xTaskCreate(camera_task, "camera_task", 6144, NULL, 6, NULL);
}