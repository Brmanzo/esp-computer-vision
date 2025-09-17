// wifi_cam.c
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

static const char *TAG = "wifi_cam";

/* -------------------- Internal state -------------------- */
static httpd_handle_t      s_httpd      = NULL;
static SemaphoreHandle_t   s_img_mutex  = NULL;
static uint8_t            *s_last_buf   = NULL;
static size_t              s_last_len   = 0;
static const char         *s_last_mime  = "application/octet-stream"; // "image/jpeg" or "image/bmp"
static bool                s_started    = false;

/* -------------------- Helpers: build 8-bit GRAY BMP -------------------- */
// Little-endian writers
static inline void le16(uint8_t *p, uint16_t v){ p[0]=v; p[1]=v>>8; }
static inline void le32(uint8_t *p, uint32_t v){ p[0]=v; p[1]=v>>8; p[2]=v>>16; p[3]=v>>24; }

/*
 * Build an 8-bit grayscale BMP with a 256-entry grayscale palette.
 * Top-down orientation (negative height) so we can dump buffer as-is.
 * Rows are padded to 4 bytes per BMP spec.
 * Returns malloc'd buffer you must free.
 */
static uint8_t *build_gray8_bmp(const uint8_t *gray, uint16_t W, uint16_t H, size_t *out_len)
{
    // Row stride must be padded to 4 bytes
    uint32_t stride = (W + 3) & ~3U;
    uint32_t img_bytes = stride * H;

    const uint32_t file_hdr = 14;
    const uint32_t dib_hdr  = 40;
    const uint32_t palette  = 256 * 4; // BGRA
    const uint32_t offBits  = file_hdr + dib_hdr + palette;
    const uint32_t file_sz  = offBits + img_bytes;

    uint8_t *bmp = (uint8_t*)malloc(file_sz);
    if (!bmp) return NULL;

    // ----- FILE HEADER (14) -----
    memset(bmp, 0, file_sz);
    bmp[0]='B'; bmp[1]='M';
    le32(&bmp[ 2], file_sz);
    le32(&bmp[10], offBits);

    // ----- DIB (BITMAPINFOHEADER, 40) -----
    uint8_t *dib = bmp + 14;
    le32(&dib[ 0], 40);             // header size
    le32(&dib[ 4], W);              // width
    le32(&dib[ 8], (uint32_t)(-(int32_t)H)); // height (negative = top-down)
    le16(&dib[12], 1);              // planes
    le16(&dib[14], 8);              // bpp
    le32(&dib[16], 0);              // BI_RGB
    le32(&dib[20], img_bytes);      // image size
    // ppm fields can be zero
    le32(&dib[32], 256);            // colors used

    // ----- PALETTE (256 * BGRA) -----
    uint8_t *pal = bmp + offBits - palette;
    for (int i=0;i<256;i++){
        pal[i*4 + 0] = i; // B
        pal[i*4 + 1] = i; // G
        pal[i*4 + 2] = i; // R
        pal[i*4 + 3] = 0; // A
    }

    // ----- PIXELS -----
    uint8_t *dst = bmp + offBits;
    for (uint32_t y=0; y<H; ++y) {
        // copy one row
        memcpy(dst, gray + (size_t)y*W, W);
        // pad to stride
        if (stride > W) memset(dst + W, 0, stride - W);
        dst += stride;
    }

    if (out_len) *out_len = file_sz;
    return bmp;
}

/* -------------------- HTTP handlers -------------------- */
static esp_err_t frame_handler(httpd_req_t *req) {
    xSemaphoreTake(s_img_mutex, portMAX_DELAY);
    const uint8_t *buf = s_last_buf;
    size_t len = s_last_len;
    const char *mime = s_last_mime;
    xSemaphoreGive(s_img_mutex);

    if (!buf || !len) {
        httpd_resp_send_404(req);
        return ESP_OK;
    }

    httpd_resp_set_type(req, mime);
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    return httpd_resp_send(req, (const char*)buf, len);
}

static const char INDEX_HTML[] =
"<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>"
"<style>body{margin:0;background:#111;display:grid;place-items:center;height:100vh}"
"img{max-width:100vw;max-height:100vh;image-rendering:-webkit-optimize-contrast}</style>"
"<img id=i src=/frame>"
"<script>setInterval(()=>{i.src='/frame?'+Date.now()},1000)</script>";

static esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    return httpd_resp_send(req, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

static httpd_handle_t start_webserver(void) {
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.lru_purge_enable = true;   // reclaim handlers if OOM
    httpd_handle_t srv = NULL;
    if (httpd_start(&srv, &cfg) == ESP_OK) {
        httpd_uri_t u_idx   = { .uri="/",     .method=HTTP_GET, .handler=index_handler };
        httpd_uri_t u_frame = { .uri="/frame",.method=HTTP_GET, .handler=frame_handler };
        httpd_uri_t u_jpg   = { .uri="/jpg",  .method=HTTP_GET, .handler=frame_handler }; // alias
        httpd_register_uri_handler(srv, &u_idx);
        httpd_register_uri_handler(srv, &u_frame);
        httpd_register_uri_handler(srv, &u_jpg);
    }
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

    if (!s_img_mutex) {
        s_img_mutex = xSemaphoreCreateMutex();
        if (!s_img_mutex) {
            ESP_LOGE(TAG, "Failed to create image mutex");
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

/* Publish a JPEG (keeps backward-compatibility with your existing flow) */
void wifi_cam_publish_jpeg(const uint8_t *jpeg, size_t len) {
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

    xSemaphoreTake(s_img_mutex, portMAX_DELAY);
    if (s_last_buf) free(s_last_buf);
    s_last_buf  = copy;
    s_last_len  = len;
    s_last_mime = "image/jpeg";
    xSemaphoreGive(s_img_mutex);
}

/* Old name kept as alias */
void wifi_cam_publish(const uint8_t *jpeg, size_t len) {
    wifi_cam_publish_jpeg(jpeg, len);
}

/* Publish a grayscale 8-bit image as a BMP */
void wifi_cam_publish_gray8_as_bmp(const uint8_t *gray, uint16_t w, uint16_t h) {
    if (!gray || w == 0 || h == 0 || !s_started) return;

    size_t bmp_len = 0;
    uint8_t *bmp = build_gray8_bmp(gray, w, h, &bmp_len);
    if (!bmp) {
        ESP_LOGW(TAG, "Failed to build BMP (%ux%u)", (unsigned)w, (unsigned)h);
        return;
    }

    xSemaphoreTake(s_img_mutex, portMAX_DELAY);
    if (s_last_buf) free(s_last_buf);
    s_last_buf  = bmp;
    s_last_len  = bmp_len;
    s_last_mime = "image/bmp";
    xSemaphoreGive(s_img_mutex);
}