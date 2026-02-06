// gpio.h
// Bradley Manzo 2026

// Free                     GPIO_NUM_0
#define UART_RTS_PIN        GPIO_NUM_1
#define GPIO_RESET_FPGA     GPIO_NUM_2
#define GPIO_BYPASS_FPGA    GPIO_NUM_3

#define PIN_SCK             GPIO_NUM_4
#define PIN_MISO            GPIO_NUM_5
#define PIN_MOSI            GPIO_NUM_6
#define PIN_CS              GPIO_NUM_7

#define PIN_SCL             GPIO_NUM_8
// Boot Button              GPIO_NUM_9
#define PIN_SDA             GPIO_NUM_10

// USB D-                   GPIO_NUM_18
// USB D+                   GPIO_NUM_19

#define RXD_PIN             GPIO_NUM_20
#define TXD_PIN             GPIO_NUM_21

void gpio_init(void);

void fpga_reset(void);