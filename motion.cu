#include "c63.h"
#include "motion.h"
#include <inttypes.h>

__global__ void gpu_mc_block_8x8_YUV(uint8_t* predicted_Y, uint8_t* recons_Y,
		struct macroblock *mbs_Y, int w, int cols, int rows)
{
	int tid_x;
	int tid_y;

	if(blockIdx.x < cols && blockIdx.y < rows)
	{
		tid_x = blockIdx.x * 8 + threadIdx.x;
		tid_y = blockIdx.y * 8 + threadIdx.y;

		int mb_number = blockIdx.y * w / 8 + blockIdx.x;
		struct macroblock mb = mbs_Y[mb_number];

		if (!mb.use_mv)
			return;

		predicted_Y[tid_y*w+tid_x] = recons_Y[(tid_y + mb.mv_y) * w + (tid_x + mb.mv_x)];
	}
}

__host__ void gpu_c63_motion_compensate(uint8_t *d_predicted_Y, uint8_t *d_predicted_U, uint8_t *d_predicted_V,
		uint8_t *d_ref_recons_Y, uint8_t *d_ref_recons_U, uint8_t *d_ref_recons_V,
		struct macroblock *d_mbs_Y, struct macroblock *d_mbs_U, struct macroblock *d_mbs_V,
		c63_common_gpu *gpu_cm ,cudaStream_t *streams)
{
	const dim3 threadsPerBlock(8, 8);
	const dim3 numBlocks(gpu_cm->ypw / threadsPerBlock.x, gpu_cm->yph / threadsPerBlock.y);
	const dim3 numBlocks2(gpu_cm->upw / threadsPerBlock.x, gpu_cm->uph / threadsPerBlock.y);
	const dim3 numBlocks3(gpu_cm->vpw / threadsPerBlock.x, gpu_cm->vph / threadsPerBlock.y);

	gpu_mc_block_8x8_YUV<<<numBlocks, threadsPerBlock, 0, streams[0]>>>
	(d_predicted_Y, d_ref_recons_Y, d_mbs_Y, gpu_cm->width, gpu_cm->mb_cols, gpu_cm->mb_rows);

	gpu_mc_block_8x8_YUV<<<numBlocks2, threadsPerBlock, 0, streams[1]>>>
	(d_predicted_U, d_ref_recons_U, d_mbs_U, gpu_cm->width/2, gpu_cm->mb_cols/2, gpu_cm->mb_rows/2);

	gpu_mc_block_8x8_YUV<<<numBlocks3, threadsPerBlock, 0, streams[2]>>>
	(d_predicted_V, d_ref_recons_V, d_mbs_V, gpu_cm->width/2, gpu_cm->mb_cols/2, gpu_cm->mb_rows/2);
}









