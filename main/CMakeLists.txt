idf_component_register(
    SRCS "arducam.c" "arducam_demo.c" "wifi_cam.c"
    INCLUDE_DIRS "."
    REQUIRES esp_wifi esp_http_server esp_netif nvs_flash
    PRIV_REQUIRES esp_driver_i2c esp_driver_spi esp_driver_gpio esp_driver_uart
)