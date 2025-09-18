#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "driver/uart.h"

#include "includes/arducam.h"

/* -------------------------------------- UART -------------------------------------- */
/* Initialize UART for streaming images to interfaces. */
static uint8_t cameraCommand;
static QueueHandle_t uart_queue_handle;

void uart_init(void) {
    uart_config_t uart_config  = {};
        uart_config.baud_rate  = BAUD_RATE;
        uart_config.data_bits  = UART_DATA_8_BITS;
        uart_config.parity     = UART_PARITY_DISABLE;
        uart_config.stop_bits  = UART_STOP_BITS_1;
        uart_config.flow_ctrl  = UART_HW_FLOWCTRL_DISABLE;
        uart_config.source_clk = UART_SCLK_DEFAULT;

    // Install driver and ask it to create an event queue of 20 items
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, RX_BUF_SIZE, 0, QUEUE_DEPTH, &uart_queue_handle, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM, UART_TX_PIN, UART_RX_PIN,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

/* Task to handle UART events, such as receiving data. */
void uart_event_task(void *pvParameters) { // uart/events/example
    uart_event_t event;
    uint8_t* dtmp = (uint8_t*) malloc(RD_BUF_SIZE);
    
    for(;;) {
        // Wait for the next event in the queue
        if(xQueueReceive(uart_queue_handle, (void * )&event, (TickType_t)portMAX_DELAY)) {
            bzero(dtmp, RD_BUF_SIZE);
            switch(event.type) {
                // Event when UART data is received
                case UART_DATA:
                ESP_LOGI("UART_HANDLER", "[UART DATA]: %d", event.size);
                    // Read the received data from the UART buffer
                    int len = uart_read_bytes(UART_NUM, dtmp, event.size, portMAX_DELAY);
                    
                    // Echo the data back to the sender
                    uart_write_bytes(UART_NUM, (const char*) dtmp, len);

                    if (len > 0) {
                        cameraCommand = dtmp[0];
                        ESP_LOGI("UART_HANDLER", "Received new command: %c", cameraCommand);
                    }
                    break;
                
                // Other event types can be handled here
                default:
                    // Log other event types
                    ESP_LOGI("UART_HANDLER", "uart event type: %d", event.type);
                    break;
            }
        }
    }
    free(dtmp);
    dtmp = NULL;
    vTaskDelete(NULL);
}