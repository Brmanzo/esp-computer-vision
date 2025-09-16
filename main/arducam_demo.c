// app_main.c
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#include "arducam.h"     // arducam_yuv_init(), arducam_camlock_take/give(), uart_event_task()
#include "wifi_cam.h"    // wifi_cam_init(), wifi_cam_set_frame_dims()

static const char *TAG = "main";

// Match your YUV init (you said QVGA)
#define FRAME_W  320
#define FRAME_H  240

// If you ever want the old JPEG loop back, flip this to 1 and make sure it uses the cam lock.
#define START_JPEG_TASK  0

#if START_JPEG_TASK
extern void singleCapture(void);
static void camera_task(void *arg) {
    for (;;) {
        arducam_camlock_take();
        singleCapture();           // publishes /jpg
        arducam_camlock_give();
        vTaskDelay(pdMS_TO_TICKS(800));
    }
}
#endif

// Provided by your code already
extern void uart_event_task(void *arg);

// --- main ---
void app_main(void)
{
    ESP_LOGI(TAG, "Calling app_main()");

    // 1) Full camera bring-up + OV2640 configured for YUV422 (your combined pipeline)
    //    This should: init SPI/I2C/UART, power/reset sensor via ArduCHIP GPIO, probe ID,
    //    and apply your YUV QVGA register table. Do NOT leave the sensor in JPEG mode.
    arducam_yuv_init();
    ESP_LOGI(TAG, "YUV init done.");

    // 2) (optional) start your UART event task, if you still want console commands/echo
    xTaskCreate(uart_event_task, "uart_event_task", 4096, NULL, 5, NULL);

    // 3) Start SoftAP + HTTP server. Pages: '/', '/jpg', '/gray'
    ESP_ERROR_CHECK(wifi_cam_init("esp-cam", "12345678"));
    // Tell /gray the frame size your sensor is outputting
    wifi_cam_set_frame_dims(FRAME_W, FRAME_H);

#if START_JPEG_TASK
    // 4) (optional) legacy JPEG loop — generally DISABLE while testing YUV
    xTaskCreate(camera_task, "camera_task", 6144, NULL, 6, NULL);
#else
    ESP_LOGI(TAG, "JPEG loop disabled; use http://<ap-ip>/gray for live grayscale.");
#endif
}