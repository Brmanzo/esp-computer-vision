// arducam.h
// Bradley Manzo 2026
#ifndef ARDUCAM_H_ 
#define ARDUCAM_H_

#include "driver/gpio.h"
#include "esp_err.h"

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
#define I2C_TIMEOUT_MS             1000
#define WRITE_BIT                  0x80
#define ARDUCAM_ADDR               0x30

/* uart pin source */
#define UART_NUM    UART_NUM_0
#define BAUD_RATE   921600
#define DATA_BITS   8
#define STOP_BITS   1
#define PARITY      UART_PARITY_NONE
#define UART_TX_PIN GPIO_NUM_0
#define UART_RX_PIN GPIO_NUM_1
#define BUF_SIZE    (256)
#define QUEUE_DEPTH (20)
#define RX_BUF_SIZE (BUF_SIZE*2)
#define RD_BUF_SIZE (BUF_SIZE)

/* jpeg markers*/
#define MARKER_PREFIX 0xFF
#define SOI           0xD8
#define EOI           0xD9

/* camera registers */
#define ARDUCHIP_MODE 0x02
#define CAM2LCD_MODE  0x01  // JPEG path

#ifndef _SENSOR_
#define _SENSOR_
    struct sensor_reg {
        unsigned int reg;
        unsigned int val;
    };
#endif

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
#define ARDUCHIP_TEST1          0x00
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
#define RECALIBRATE_INTERVAL      20

#define FIFO_SIZE1				0x42  //Camera write FIFO size[7:0] for burst to read
#define FIFO_SIZE2				0x43  //Camera write FIFO size[15:8]
#define FIFO_SIZE3				0x44  //Camera write FIFO size[18:16]

#define RJPEG_PULL_CHUNK   1024   // bytes per SPI xfer (512..2048 are reasonable)
#define RJPEG_YIELD_EVERY  1      // yield after this many chunks (1 = yield each chunk)

// Scale constants aren't always defined — cast integers instead.
#ifndef JPEG_IMAGE_SCALE_0
#  define JPEG_IMAGE_SCALE_0 ((esp_jpeg_image_scale_t)0)  // 1/1
#endif
#ifndef JPEG_IMAGE_SCALE_1
#  define JPEG_IMAGE_SCALE_1 ((esp_jpeg_image_scale_t)1)  // 1/2
#endif
#ifndef JPEG_IMAGE_SCALE_2
#  define JPEG_IMAGE_SCALE_2 ((esp_jpeg_image_scale_t)2)  // 1/4
#endif
#ifndef JPEG_IMAGE_SCALE_3
#  define JPEG_IMAGE_SCALE_3 ((esp_jpeg_image_scale_t)3)  // 1/8
#endif

// Not all builds expose a gray output format. Fallback to RGB565.
#ifndef JPEG_IMAGE_FORMAT_GRAY
#  define JPEG_IMAGE_FORMAT_GRAY JPEG_IMAGE_FORMAT_RGB565
#  define JPEG_NEEDS_GRAY_FALLBACK 1
#else
#  define JPEG_NEEDS_GRAY_FALLBACK 0
#endif

extern struct camera_operate arducam;

/* Take control of arducam hardware. */
void arducam_camlock_take(void);

/* Release arducam hardware. */
void arducam_camlock_give(void);

/* Reset the contents and state of the Arducam FIFO Buffer. */
void arducam_reset_fifo(void);

/* Set Arducam to capture mode. */
void arducam_set_capture(void);

/* Start the image capture. */
void arducam_start_capture(void);

/* Stop the image capture. */
void arducam_stop_capture(void);

/* Read raw YUV422 image data from Arducam and pack luma values onto 1bpp local buffer. */
esp_err_t arducam_read_and_pack_stream(uint8_t *out, size_t out_cap, uint16_t w, uint16_t h, uint8_t* adaptive_th, uint8_t capture_num);

#endif