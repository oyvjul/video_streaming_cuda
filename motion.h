#include <inttypes.h>
#include "c63.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void gpu_c63_motion_compensate(uint8_t *d_predicted, uint8_t *d_predictedU, uint8_t *d_predictedV,
		uint8_t *d_ref_reconsY, uint8_t *d_ref_reconsU, uint8_t *d_ref_reconsV,
		struct macroblock *d_mbsY, struct macroblock *d_mbsU, struct macroblock *d_mbsV,
		c63_common_gpu *gpu_cm ,cudaStream_t *streams);
