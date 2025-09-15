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
    arducam.systemInit();                      // your I2C/SPI/UART + power-up
    if (arducam.busDetect() != 0) return;
    if (arducam.cameraProbe() != 0) return;

    ov2640_attach();                           // <-- new
    ov2640_configure_yuv_qvga();               // or ov2640_configure_jpeg_qvga();

    wifi_cam_init("esp-cam","12345678");
    xTaskCreate(camera_task, "camera_task", 6144, NULL, 6, NULL);
}