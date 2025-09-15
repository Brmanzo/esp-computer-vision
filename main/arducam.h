#ifndef ARDUCAM_H_ // "Guard" starts here
#define ARDUCAM_H_

#include <stdio.h>
#include <string.h>
#include "sdkconfig.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/i2c_master.h"

/*spi pin source*/
#define PIN_SCK    GPIO_NUM_4
#define PIN_MISO   GPIO_NUM_5
#define PIN_MOSI   GPIO_NUM_6
#define PIN_CS     GPIO_NUM_7
#define SPI_CHUNK  4092
#define SPI_FREQ   8000000

/*i2c pin source */
#define PIN_SCL                    GPIO_NUM_8
#define PIN_SDA                    GPIO_NUM_10
#define I2C_MASTER_NUM             I2C_NUM_0
#define I2C_MASTER_FREQ_HZ         400000
#define I2C_MASTER_TX_BUF_DISABLE  0
#define I2C_MASTER_RX_BUF_DISABLE  0
#define I2C_TIMEOUT_MS             1000
#define WRITE_BIT                  0x80

/* uart pin source */
#define UART_NUM UART_NUM_0
#define BAUD_RATE 921600
#define DATA_BITS 8
#define STOP_BITS 1
#define PARITY    UART_PARITY_NONE
#define UART_TX_PIN GPIO_NUM_0
#define UART_RX_PIN GPIO_NUM_1
#define BUF_SIZE (256)
#define QUEUE_DEPTH (20)
#define RX_BUF_SIZE (BUF_SIZE*2)
#define RD_BUF_SIZE (BUF_SIZE)

typedef struct sensor_reg {
    unsigned int reg;
    unsigned int val;
} sensor_reg;

struct sensor_info{
    uint8_t sensor_slave_address;
    uint8_t address_size;
    uint8_t data_size; 
    uint16_t sensor_id;
};
struct camera_operate{
    uint8_t slave_address;
    void (*systemInit)(void);
    uint8_t (*busDetect) (void);
    uint8_t (*cameraProbe) (void);
    void  (*cameraInit) (void);
    void (*setJpegSize)(uint8_t size);
};
#define res_160x120 		0	//160x120
#define res_176x144 		1	//176x144
#define res_320x240 		2	//320x240
#define res_352x288 		3	//352x288
#define res_640x480		    4	//640x480
#define res_800x600 		5	//800x600
#define res_1024x768		6	//1024x768
#define res_1280x1024	7	//1280x1024
#define res_1600x1200	8	//1600x1200
#define ARDUCHIP_FIFO      		0x04  //FIFO and I2C control
#define FIFO_CLEAR_MASK    		0x01
#define FIFO_START_MASK    		0x02
#define FIFO_RDPTR_RST_MASK     0x10
#define FIFO_WRPTR_RST_MASK     0x20
#define ARDUCHIP_GPIO			0x06  //GPIO Write Register
#define GPIO_RESET_MASK			0x01  //0 = Sensor reset,							1 =  Sensor normal operation
#define GPIO_PWDN_MASK			0x02  //0 = Sensor normal operation, 	1 = Sensor standby
#define GPIO_PWREN_MASK			0x04	//0 = Sensor LDO disable, 			1 = sensor LDO enable

#define BURST_FIFO_READ			0x3C  //Burst FIFO read operation
#define SINGLE_FIFO_READ		0x3D  //Single FIFO read operation

#define ARDUCHIP_REV       		0x40  //ArduCHIP revision
#define VER_LOW_MASK       		0x3F
#define VER_HIGH_MASK      		0xC0

#define ARDUCHIP_TRIG      		0x41  //Trigger source
#define VSYNC_MASK         		0x01
#define SHUTTER_MASK       		0x02
#define CAP_DONE_MASK      		0x08

#define FIFO_SIZE1				0x42  //Camera write FIFO size[7:0] for burst to read
#define FIFO_SIZE2				0x43  //Camera write FIFO size[15:8]
#define FIFO_SIZE3				0x44  //Camera write FIFO size[18:16]

extern volatile uint8_t cameraCommand;
extern struct camera_operate arducam;
extern uint8_t slave_addr;
int rdSensorReg8_8(uint8_t regID, uint8_t* regDat );
int wrSensorReg8_8(uint8_t regID, uint8_t regDat );
int wrSensorRegs8_8(const struct sensor_reg reglist[]);
void write_reg(uint8_t address, uint8_t value);
uint8_t read_reg(uint8_t address);
void singleCapture(void);
void uart_event_task(void *pvParameters);
void esp32c3_SystemInit(void);
void ov2640_attach(void);
void ov2640_configure_yuv_qvga(void);
void ov2640_configure_jpeg_qvga(void);
#endif