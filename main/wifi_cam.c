#include "wifi_cam.h"

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

// We call into your ArduCAM-side streaming helper for grayscale BMP.
#include "arducam.h"   // must declare: esp_err_t arducam_stream_gray_bmp(httpd_req_t*, uint16_t, uint16_t);

static const char *TAG = "wifi_cam";

/* -------------------- Internal state -------------------- */
static httpd_handle_t      s_httpd     = NULL;
static SemaphoreHandle_t   s_jpg_mutex = NULL;
static uint8_t            *s_last_jpg  = NULL;
static size_t              s_last_len  = 0;
static bool                s_started   = false;

// Frame size used by /gray (YUV->grayscale). Defaults to QVGA.
static uint16_t s_frame_w = 320;
static uint16_t s_frame_h = 240;

/* -------------------- HTTP handlers -------------------- */
static esp_err_t jpg_handler(httpd_req_t *req) {
    if (!s_started) return httpd_resp_send_500(req);

    xSemaphoreTake(s_jpg_mutex, portMAX_DELAY);
    const uint8_t *buf = s_last_jpg;
    const size_t   len = s_last_len;
    xSemaphoreGive(s_jpg_mutex);

    if (!buf || !len) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    return httpd_resp_send(req, (const char*)buf, len);
}

static esp_err_t gray_handler(httpd_req_t *req) {
    if (!s_started) return httpd_resp_send_500(req);
    // Streams a top-down 8-bit BMP created from Y plane of current YUV422 frame.
    // Make sure your camera is currently outputting YUV422.
    return arducam_stream_gray_bmp(req, s_frame_w, s_frame_h);
}

static const char INDEX_HTML[] =
"<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>"
"<style>"
"html,body{height:100%;margin:0;background:#0b0b0b;color:#ddd;font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial}"
".wrap{display:grid;grid-template-rows:auto 1fr;gap:12px;height:100%;}"
"header{display:flex;gap:8px;align-items:center;justify-content:center;padding:10px;background:#111;box-shadow:0 2px 6px #0008}"
"button{background:#1e1e1e;border:1px solid #333;color:#ddd;padding:8px 12px;border-radius:10px;cursor:pointer}"
"button:hover{background:#252525}"
"main{display:grid;place-items:center}"
"img{max-width:95vw;max-height:80vh;image-rendering:pixelated;outline:1px solid #222;border-radius:8px}"
"</style>"
"<div class=wrap>"
"<header>"
" <button onclick=\"setSrc('jpg')\">JPEG</button>"
" <button onclick=\"setSrc('gray')\">Grayscale</button>"
" <span style='opacity:.7;padding-left:10px'>/jpg is your previous pipeline; /gray renders the Y plane as an 8-bit BMP.</span>"
"</header>"
"<main><img id=i src='/jpg'></main>"
"</div>"
"<script>"
"let mode='jpg';"
"function setSrc(m){mode=m;refresh()}"
"function refresh(){document.getElementById('i').src='/' + mode + '?' + Date.now()}"
"setInterval(refresh, 1000);"
"</script>";

static esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    return httpd_resp_send(req, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t no_content_handler(httpd_req_t *req) {
    httpd_resp_set_status(req, "204 No Content");
    return httpd_resp_send(req, NULL, 0);
}


static httpd_handle_t start_webserver(void) {
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.lru_purge_enable = true;   // reclaim handlers if OOM

    httpd_handle_t srv = NULL;
    if (httpd_start(&srv, &cfg) != ESP_OK) return NULL;

    httpd_uri_t u_idx  = { .uri="/",     .method=HTTP_GET, .handler=index_handler };
    httpd_uri_t u_jpg  = { .uri="/jpg",  .method=HTTP_GET, .handler=jpg_handler   };
    httpd_uri_t u_gray = { .uri="/gray", .method=HTTP_GET, .handler=gray_handler  };
    httpd_uri_t u_fav   = { .uri="/favicon.ico",         .method=HTTP_GET, .handler=no_content_handler };
    httpd_uri_t u_apple = { .uri="/apple-touch-icon.png", .method=HTTP_GET, .handler=no_content_handler };
    httpd_uri_t u_robot = { .uri="/robots.txt",           .method=HTTP_GET, .handler=no_content_handler };
    httpd_register_uri_handler(srv, &u_fav);
    httpd_register_uri_handler(srv, &u_apple);
    httpd_register_uri_handler(srv, &u_robot);

    httpd_register_uri_handler(srv, &u_idx);
    httpd_register_uri_handler(srv, &u_jpg);
    httpd_register_uri_handler(srv, &u_gray);

    return srv;
}

/* -------------------- Wi-Fi helpers -------------------- */

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
        ESP_LOGI(TAG, "STA connected: " MACSTR, MAC2STR(e->mac));
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t *e = (wifi_event_ap_stadisconnected_t*)data;
        ESP_LOGI(TAG, "STA disconnected: " MACSTR, MAC2STR(e->mac));
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
    const char *S = ssid ? ssid : "esp-cam";
    const char *P = pass ? pass : "12345678";

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

/* -------------------- Public API -------------------- */

esp_err_t wifi_cam_init(const char *ssid, const char *pass) {
    if (s_started) return ESP_OK;

    // NVS once
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    if (!s_jpg_mutex) {
        s_jpg_mutex = xSemaphoreCreateMutex();
        if (!s_jpg_mutex) {
            ESP_LOGE(TAG, "Failed to create JPEG mutex");
            return ESP_ERR_NO_MEM;
        }
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

void wifi_cam_publish(const uint8_t *jpeg, size_t len) {
    if (!jpeg || !len || !s_started) return;
    if (len > WIFI_CAM_MAX_JPEG) {
        ESP_LOGW(TAG, "JPEG too large (%u > %u)", (unsigned)len, (unsigned)WIFI_CAM_MAX_JPEG);
        return;
    }

    uint8_t *copy = (uint8_t*)malloc(len);
    if (!copy) {
        ESP_LOGW(TAG, "No heap to copy JPEG (%u bytes)", (unsigned)len);
        return;
    }
    memcpy(copy, jpeg, len);

    xSemaphoreTake(s_jpg_mutex, portMAX_DELAY);
    if (s_last_jpg) free(s_last_jpg);
    s_last_jpg = copy;
    s_last_len = len;
    xSemaphoreGive(s_jpg_mutex);
}

void wifi_cam_set_frame_dims(uint16_t w, uint16_t h) {
    s_frame_w = w;
    s_frame_h = h;
}

bool wifi_cam_started(void) {
    return s_started;
}