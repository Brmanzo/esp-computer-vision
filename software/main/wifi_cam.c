// wifi_cam.c
#include "includes/wifi_cam.h"

#include <string.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#include "includes/server_html.h"

static const char *TAG = "wifi_cam";

/* -------------------- Internal state -------------------- */
static httpd_handle_t      s_httpd      = NULL;
static bool                s_started    = false;
static uint8_t             *s_frame_buf = NULL;
static size_t              s_frame_len  = 0;
static bool                s_frame_available = false;
static SemaphoreHandle_t   s_frame_mux  = NULL;

/* --------------------------- Index Event --------------------------- */
/* Send html index over ESP wifi. */
static esp_err_t index_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    return httpd_resp_send(req, INDEX_HTML, INDEX_HTML_LEN);
}
/* -------------------- Frame Event -------------------- */
/* Call once at startup to initialize */
void frame_server_init(void)
{
    if (s_frame_mux == NULL) {
        s_frame_mux = xSemaphoreCreateMutex();
    }
}

static esp_err_t frame_handler(httpd_req_t *req)
{
    uint8_t *local_buf = NULL;
    size_t local_len = 0;
    esp_err_t err = ESP_OK;

    if (s_frame_mux == NULL) frame_server_init();

    if (xSemaphoreTake(s_frame_mux, pdMS_TO_TICKS(500)) != pdTRUE) {
        ESP_LOGW(TAG, "frame_handler: mutex busy");
        httpd_resp_send_err(req, 503, "Server busy");
    }

    if (!s_frame_available || s_frame_buf == NULL || s_frame_len == 0) {
        xSemaphoreGive(s_frame_mux);
        httpd_resp_send_err(req, 503, "No frame ready");
    }

    // Allocate a temporary copy while holding mutex so s_frame_buf can't be freed underneath us.
    local_len = s_frame_len;
    local_buf = malloc(local_len);
    if (!local_buf) {
        xSemaphoreGive(s_frame_mux);
        ESP_LOGE(TAG, "frame_handler: malloc failed for len=%u", (unsigned)local_len);
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    memcpy(local_buf, s_frame_buf, local_len);

    xSemaphoreGive(s_frame_mux);

    // Send the copy
    httpd_resp_set_type(req, "application/octet-stream");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");

    err = httpd_resp_send(req, (const char *)local_buf, local_len);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "httpd_resp_send failed: %d", err);
    }

    free(local_buf);
    return err;
}

esp_err_t publish_frame(const uint8_t *data, size_t len)
{
    if (!data || len == 0) return ESP_ERR_INVALID_ARG;

    if (s_frame_mux == NULL) frame_server_init();

    // allocate new buffer
    uint8_t *newbuf = malloc(len);
    if (!newbuf) {
        ESP_LOGE(TAG, "Out of memory allocating frame buffer (len=%u)", (unsigned)len);
        return ESP_ERR_NO_MEM;
    }
    memcpy(newbuf, data, len);

    // swap under mutex
    if (xSemaphoreTake(s_frame_mux, pdMS_TO_TICKS(2000)) != pdTRUE) {
        free(newbuf);
        ESP_LOGW(TAG, "publish_frame: failed to take mutex");
        return ESP_FAIL;
    }

    // free old buffer and replace
    if (s_frame_buf) {
        free(s_frame_buf);
    }
    s_frame_buf = newbuf;
    s_frame_len = len;
    s_frame_available = true;

    xSemaphoreGive(s_frame_mux);
    return ESP_OK;
}

/* -------------------- Web Server Setup -------------------- */
static httpd_handle_t start_webserver(void) {
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.lru_purge_enable = true;   // reclaim handlers if OOM
    httpd_handle_t srv = NULL;
    if (httpd_start(&srv, &cfg) == ESP_OK) {
        httpd_uri_t u_idx   = { .uri="/",     .method=HTTP_GET, .handler=index_handler };
        httpd_uri_t u_frame = { .uri="/frame",.method=HTTP_GET, .handler=frame_handler };
        httpd_register_uri_handler(srv, &u_idx);
        httpd_register_uri_handler(srv, &u_frame);
    }
    return srv;
}
// Tolerate "already initialized" states; only error on real failures.
static esp_err_t net_stack_once(void) {
    esp_err_t e = esp_netif_init();
    if (e != ESP_OK && e != ESP_ERR_INVALID_STATE) return e;
    e = esp_event_loop_create_default();
    if (e != ESP_OK && e != ESP_ERR_INVALID_STATE) return e;
    return ESP_OK;
}

static void on_wifi_event(void* arg, esp_event_base_t base, int32_t id, void* data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t *e = (wifi_event_ap_staconnected_t*)data;
        ESP_LOGI("wifi_cam", "STA connected: " MACSTR, MAC2STR(e->mac));
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t *e = (wifi_event_ap_stadisconnected_t*)data;
        ESP_LOGI("wifi_cam", "STA disconnected: " MACSTR, MAC2STR(e->mac));
    }
}

static esp_err_t wifi_init_softap(const char *ssid, const char *pass) {
    ESP_ERROR_CHECK(net_stack_once());

    esp_netif_t *ap_netif = esp_netif_create_default_wifi_ap();
    wifi_init_config_t wicfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wicfg));

    // Log STA connect/disconnect events
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &on_wifi_event, NULL, NULL));

    wifi_config_t ap = { 0 };
    const char *S = ssid ? ssid : DEFAULT_SSID;
    const char *P = pass ? pass : DEFAULT_PASS;

    strncpy((char*)ap.ap.ssid,     S, sizeof(ap.ap.ssid)-1);
    strncpy((char*)ap.ap.password, P, sizeof(ap.ap.password)-1);
    ap.ap.ssid_len       = strlen((char*)ap.ap.ssid);
    ap.ap.channel        = 1;
    ap.ap.max_connection = 2;
    ap.ap.authmode       = (strlen(P) ? WIFI_AUTH_WPA_WPA2_PSK : WIFI_AUTH_OPEN);

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));  // keep AP awake
    ESP_ERROR_CHECK(esp_wifi_start());

    // Log AP IP
    esp_netif_ip_info_t ip;
    ESP_ERROR_CHECK(esp_netif_get_ip_info(ap_netif, &ip));
    ESP_LOGI(TAG, "SoftAP '%s' up. IP: " IPSTR "  Browse: http://" IPSTR "/",
             ap.ap.ssid, IP2STR(&ip.ip), IP2STR(&ip.ip));
    return ESP_OK;
}

esp_err_t wifi_cam_init(const char *ssid, const char *pass) {
    if (s_started) return ESP_OK;

    // NVS once
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    frame_server_init();
    if (!s_frame_mux) {
        ESP_LOGE(TAG, "Failed to create frame mutex");
        return ESP_ERR_NO_MEM;
    }
    
    ESP_ERROR_CHECK(wifi_init_softap(ssid, pass));

    s_httpd = start_webserver();
    if (!s_httpd) {
        ESP_LOGE(TAG, "HTTP server failed to start");
        return ESP_FAIL;
    }

    s_started = true;
    ESP_LOGI(TAG, "HTTP server started, free heap: %u", (unsigned)esp_get_free_heap_size());
    return ESP_OK;

}
