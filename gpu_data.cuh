#ifndef GPU_DATA
#define GPU_DATA

#include"c63.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//used for debbuging
static float final_time = 0;
static cudaEvent_t start_event, end_event;


static cudaStream_t streams[3];

#endif
