#ifndef C63_ME_H_
#define C63_ME_H_

#include "c63.h"
#include <stdint.h>

// Declaration

void c63_motion_estimate_new(uint8_t *d_predicted_Y, uint8_t *d_predicted_U, uint8_t *d_predicted_V,
		uint8_t *d_ref_recons_Y, uint8_t *d_ref_recons_U, uint8_t *d_ref_recons_V,
		struct macroblock *d_mbs_Y, struct macroblock *d_mbs_U, struct macroblock *d_mbs_V,
		c63_common_gpu *gpu_cm ,cudaStream_t *streams);

#endif  /* C63_ME_H_ */
