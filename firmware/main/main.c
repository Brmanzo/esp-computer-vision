// main.c
// Bradley Manzo 2026
#include "includes/arducam.h"
#include "includes/wifi_cam.h"
#include "includes/capture.h"
#include "includes/uart.h"
#include "includes/gpio.h"

static void camera_task(void *arg) {
    fpga_reset();

    for (;;) {
        ESP_LOGI("main", "Waiting for FPGA wakeup byte...");

        while (!uart_wait_for_wakeup_byte(UART_NUM_1, pdMS_TO_TICKS(30000))) {
            ESP_LOGI("main", "FPGA wakeup byte not received; resetting FPGA...");
            fpga_reset();
        }

        ESP_LOGI("main", "FPGA awake; starting capture loop");

        while (singleCapture()) {
            vTaskDelay(pdMS_TO_TICKS(10));
        }

        ESP_LOGW("main", "Capture failed; resetting FPGA and returning to wakeup wait");
        fpga_reset();
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

void app_main(void) {
    // Board bring-up
    arducam.systemInit();
    
    gpio_init();

    // Initialize UART to Icebreaker FPGA
    uart_init();

    // Probe and initialize ArduCAM + OV2640
    if (arducam.busDetect() != 0) { ESP_LOGE("main","SPI bus test failed."); return; }
    if (arducam.cameraProbe() != 0){ ESP_LOGE("main","Camera sensor probe failed."); return; }
    arducam.cameraInit();
    arducam.setJpegSize(res_320x240);

    // Bring up SoftAP + HTTP ONCE
    ESP_ERROR_CHECK(wifi_cam_init(DEFAULT_SSID, DEFAULT_PASS));
    vTaskDelay(pdMS_TO_TICKS(10));  // grace period so AP/HTTP are ready

    // Start capture loop task (priority 6 is usually safe)
    ESP_LOGI("HEAP", "free=%u largest=%u",
    (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
    (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    xTaskCreate(camera_task, "camera_task", 6144, NULL, 4, NULL);
}