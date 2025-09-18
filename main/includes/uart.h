#ifndef UART_H
#define UART_H

#include <stdint.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------- UART -------------------------------------- */

void uart_init(void);

/* Task to handle UART events, such as receiving data. */
void uart_event_task(void *pvParameters);

#ifdef __cplusplus
}
#endif

#endif // UART_H