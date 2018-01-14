#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "me.h"

__device__ int best_sad = INT_MAX;

__global__ void sad_gpu(uint8_t *block1, uint8_t *block2, int mx, int my, int w, int h, int range, macroblock *mb)
{

	if(blockIdx.x <= mx && blockIdx.y <= my)
	{
		int idx = blockIdx.x * 8 - range + threadIdx.x + range;
	    int idy = blockIdx.y * 8 - range + threadIdx.y + range;


		if( blockIdx.x * 8 - range < 0 ) {idx = 0;}
		if( blockIdx.y * 8 - range < 0 ) {idy = 0;}
		if( idx > (w - 8) ) {idx = w - 8;}
		if( idy > (h - 8) ) {idy = h - 8;}

		__shared__ int sad;

		sad = 0;
		int sidx = blockIdx.x * 8 + threadIdx.x;
		int sidy = blockIdx.y * 8 + threadIdx.y;

		int tmp2 =idy *w + idx;

		sad += abs(block2[tmp2] - block1[sidy*w + sidx]);
		__syncthreads();

		if(sad < best_sad)
		{
			mb[blockIdx.y * w / 8 + blockIdx.x].mv_x = idx - blockIdx.x;
			mb[blockIdx.y * w / 8 + blockIdx.x].mv_y = idy - blockIdx.y;
			best_sad = sad;
		}
		mb[blockIdx.y * w / 8 + blockIdx.x].use_mv = 1;
	}
}

void c63_motion_estimate_new(uint8_t *d_origY, uint8_t *d_origU, uint8_t *d_origV,
		uint8_t *d_ref_recons_Y, uint8_t *d_ref_recons_U, uint8_t *d_ref_recons_V,
		struct macroblock *d_mbs_Y, struct macroblock *d_mbs_U, struct macroblock *d_mbs_V,
		c63_common_gpu *gpu_cm ,cudaStream_t *streams)
{
	const dim3 threadsPerBlock(8, 8);
	const dim3 numBlocks(gpu_cm->ypw / threadsPerBlock.x, gpu_cm->yph / threadsPerBlock.y);
	const dim3 numBlocks2(gpu_cm->upw / threadsPerBlock.x, gpu_cm->uph / threadsPerBlock.y);
	const dim3 numBlocks3(gpu_cm->vpw / threadsPerBlock.x, gpu_cm->vph / threadsPerBlock.y);

	sad_gpu<<<numBlocks, threadsPerBlock, 0, streams[0]>>>
	(d_origY, d_ref_recons_Y, gpu_cm->mb_cols, gpu_cm->mb_rows, gpu_cm->width, gpu_cm->height,
	 gpu_cm->me_search_range, d_mbs_Y);
	/*if (cudaDeviceSynchronize() != cudaSuccess)
	{
	   fprintf (stderr, "Cuda call failed: err %d \n", cudaDeviceSynchronize());
	}*/
	sad_gpu<<<numBlocks2, threadsPerBlock, 0, streams[1]>>>
	(d_origU, d_ref_recons_U, gpu_cm->mb_cols/2, gpu_cm->mb_rows/2,  gpu_cm->width, gpu_cm->height,
	 gpu_cm->me_search_range/2, d_mbs_U);
	/*if (cudaDeviceSynchronize() != cudaSuccess)
	{
	   fprintf (stderr, "Cuda call failed: err %d \n", cudaDeviceSynchronize());
	}*/
	sad_gpu<<<numBlocks3, threadsPerBlock, 0, streams[2]>>>
	(d_origV, d_ref_recons_V, gpu_cm->mb_cols/2, gpu_cm->mb_rows/2, gpu_cm->width, gpu_cm->height,
	 gpu_cm->me_search_range/2, d_mbs_V);
	/*if (cudaDeviceSynchronize() != cudaSuccess)
	{
	   fprintf (stderr, "Cuda call failed: err %d \n", cudaDeviceSynchronize());
	}*/
}


