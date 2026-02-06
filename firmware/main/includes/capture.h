// capture.h
// Bradley Manzo 2026
#ifndef CAPTURE_H
#define CAPTURE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Capture a single frame, loopsback from FPGA, and publish to Wifi. */
bool singleCapture(void);

#ifdef __cplusplus
}
#endif

#endif // CAPTURE_H