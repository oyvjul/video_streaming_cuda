#include <inttypes.h>
#include "c63.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void dct_test(c63_common_gpu *cm_gpu, uint8_t* d_origY, uint8_t* d_origU,
		uint8_t* d_origV,uint8_t* d_predictedY, uint8_t* d_predictedU, uint8_t* d_predictedV,
		int16_t *d_residualsYDCT, int16_t *d_residualsUDCT, int16_t *d_residualsVDCT, cudaStream_t *streams,
		int16_t* residuals_host_Y, int16_t* residuals_host_U, int16_t* residuals_host_V);

void idct_test(c63_common_gpu *cm_gpu, uint8_t* d_predictedY, uint8_t* d_predictedU, uint8_t* d_predictedV,
		int16_t *d_residualsYDCT, int16_t *d_residualsUDCT, int16_t *d_residualsVDCT,uint8_t *d_current_reconsY,
		uint8_t *d_current_reconsU, uint8_t *d_current_reconsV, cudaStream_t *streams);
