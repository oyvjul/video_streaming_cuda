#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda.h"

#include "cuda_dct.h"
#include "c63.h"
#include "gpu_data.cuh"

__device__ uint8_t GPU_zigzag_U[64] =
{
  0,
  1, 0,
  0, 1, 2,
  3, 2, 1, 0,
  0, 1, 2, 3, 4,
  5, 4, 3, 2, 1, 0,
  0, 1, 2, 3, 4, 5, 6,
  7, 6, 5, 4, 3, 2, 1, 0,
  1, 2, 3, 4, 5, 6, 7,
  7, 6, 5, 4, 3, 2,
  3, 4, 5, 6, 7,
  7, 6, 5, 4,
  5, 6, 7,
  7, 6,
  7,
};

__device__ uint8_t GPU_zigzag_UV[64] =
{
  0,
  0, 1,
  2, 1, 0,
  0, 1, 2, 3,
  4, 3, 2, 1, 0,
  0, 1, 2, 3, 4, 5,
  6, 5, 4, 3, 2, 1, 0,
  0, 1, 2, 3, 4, 5, 6, 7,
  7, 6, 5, 4, 3, 2, 1,
  2, 3, 4, 5, 6, 7,
  7, 6, 5, 4, 3,
  4, 5, 6, 7,
  7, 6, 5,
  6, 7,
  7,
};

__constant__ float gpu_dctlookup[8][8] =
{
  {1.0f,  0.980785f,  0.923880f,  0.831470f,  0.707107f,  0.555570f,  0.382683f,  0.195090f, },
  {1.0f,  0.831470f,  0.382683f, -0.195090f, -0.707107f, -0.980785f, -0.923880f, -0.555570f, },
  {1.0f,  0.555570f, -0.382683f, -0.980785f, -0.707107f,  0.195090f,  0.923880f,  0.831470f, },
  {1.0f,  0.195090f, -0.923880f, -0.555570f,  0.707107f,  0.831470f, -0.382683f, -0.980785f, },
  {1.0f, -0.195090f, -0.923880f,  0.555570f,  0.707107f, -0.831470f, -0.382683f,  0.980785f, },
  {1.0f, -0.555570f, -0.382683f,  0.980785f, -0.707107f, -0.195090f,  0.923880f, -0.831470f, },
  {1.0f, -0.831470f,  0.382683f,  0.195090f, -0.707107f,  0.980785f, -0.923880f,  0.555570f, },
  {1.0f, -0.980785f,  0.923880f, -0.831470f,  0.707107f, -0.555570f,  0.382683f, -0.195090f, },
};

//only need 1 for Y and 1 for U & V
// divided elements with /2.5
__device__ uint8_t gpu_quanttbl1D[2][64] =
{
{
	//Y
	6,  4,  4,  5,  4,  4,  6,  5,
	5,  5,  7,  6,  6,  7,  9,  16,
	10,  9,  8,  8,  9,  19,  14,  14,
	11,  16,  23,  20,  24,  12,  22,  20,
	22,  22,  25,  28,  36,  31,  25,  27,
	34,  27,  22,  22,  32,  43,  32,  34,
	38,  39,  41,  41,  41,  24,  30,  45,
	48,  44,  40,  48,  36,  40,  41,  39,
	},
	// U, V
	{
	6,  7,  7,  9,  8,  9,  18,  10,
	10,  18,  39,  26,  22,  26,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	},
};

__constant__ float GPU_ISQRT2 = 0.70710678118654f;
__constant__ float GPU_SCALE_VALUE = 1.0f;
__constant__ float GPU_QUANT_VALUE = 4.0f;
__shared__ float a1;
__shared__ float a2;
__shared__ float dct;

__device__ static void gpu_quantize_block(float *in_data, float *out_data, int index)
{
	int tid = threadIdx.y * 8 + threadIdx.x;

    uint8_t u = GPU_zigzag_U[tid];
    uint8_t v = GPU_zigzag_UV[tid];

	dct = in_data[v*8+u];

    // passing threads in a 2-dimensional way in a 1-dimensional array
	out_data[tid] = (float) roundf((dct /  GPU_QUANT_VALUE) / gpu_quanttbl1D[index][tid]);
}

__device__ void gpu_dct_quant_block_8x8(uint8_t *in_data, uint8_t *prediction, int w, int16_t *out_data, int index)
{
	int tid_x = threadIdx.x;
  	int tid_y = threadIdx.y;
	int tid = tid_y * 8 + tid_x;
	int i;

	__shared__ float dct_in[8*8];
	__shared__ float dct_out[8*8];

    dct_in[tid_y*8+tid_x] = ((float)in_data[tid_y*w+tid_x] - prediction[tid_y*w+tid_x]);
    //quantize operation
    dct = 0;

    for (i = 0; i < 8; ++i)
    	dct += dct_in[i+tid_y*8] * gpu_dctlookup[i][tid_x];

    dct_out[tid_x+tid_y*8] = dct;
	__syncthreads();

	//transpose
	dct_in[tid_y*8+tid_x] = dct_out[tid_x*8+tid_y];

	__syncthreads();

	//quantize operation
	dct = 0;

	for (i = 0; i < 8; ++i)
	    dct += dct_in[i+tid_y*8] * gpu_dctlookup[i][tid_x];

	dct_out[tid_x+tid_y*8] = dct;
	__syncthreads();
	//transpose
	dct_in[tid_y*8+tid_x] = dct_out[tid_x*8+tid_y];

	__syncthreads();

	//scaling
	a1 = !tid_x ? GPU_ISQRT2 : GPU_SCALE_VALUE;
	a2 = !tid_y ? GPU_ISQRT2 : GPU_SCALE_VALUE;

	dct_out[tid_y*8+tid_x] = dct_in[tid_y*8+tid_x] * a1 * a2;

	__syncthreads();

	gpu_quantize_block(dct_out, dct_in, index);

	out_data[tid] = dct_in[tid];
}

__global__ void gpu_dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width, int16_t *out_data, int index)
{
	int bid = blockIdx.y * (width / 8) + blockIdx.x;
	int bid_x = blockIdx.x * 8;
	int bid_y = blockIdx.y * width * 8;
	int offset =  bid_x+bid_y;

	gpu_dct_quant_block_8x8(in_data+offset, prediction+offset, width, out_data+(bid*64), index);
}

__host__ void dct_test(c63_common_gpu *cm_gpu, uint8_t* d_origY, uint8_t* d_origU,
		uint8_t* d_origV,uint8_t* d_predictedY, uint8_t* d_predictedU, uint8_t* d_predictedV,
		int16_t *d_residualsYDCT, int16_t *d_residualsUDCT, int16_t *d_residualsVDCT, cudaStream_t *streams,
		int16_t* residuals_host_Y, int16_t* residuals_host_U, int16_t* residuals_host_V)
{
	const dim3 threadsPerBlock(8, 8);
	const dim3 numBlocks(cm_gpu->ypw / threadsPerBlock.x, cm_gpu->yph / threadsPerBlock.y);
	const dim3 numBlocks2(cm_gpu->upw / threadsPerBlock.x, cm_gpu->uph / threadsPerBlock.y);
	const dim3 numBlocks3(cm_gpu->vpw / threadsPerBlock.x, cm_gpu->vph / threadsPerBlock.y);


	gpu_dct_quantize<<<numBlocks, threadsPerBlock, 0, streams[0]>>>
		(d_origY, d_predictedY, cm_gpu->ypw, d_residualsYDCT, 0);
	cudaMemcpyAsync(residuals_host_Y, d_residualsYDCT,
			cm_gpu->ypw * cm_gpu->yph * sizeof(int16_t), cudaMemcpyDeviceToHost,
			streams[0]);

	gpu_dct_quantize<<<numBlocks2, threadsPerBlock, 0, streams[1]>>>
		(d_origU, d_predictedU, cm_gpu->upw, d_residualsUDCT, 1);
	cudaMemcpyAsync(residuals_host_U, d_residualsUDCT,
			cm_gpu->upw * cm_gpu->uph * sizeof(int16_t), cudaMemcpyDeviceToHost,
			streams[1]);

	gpu_dct_quantize<<<numBlocks3, threadsPerBlock, 0, streams[2]>>>
			(d_origV, d_predictedV, cm_gpu->vpw, d_residualsVDCT, 1);
	cudaMemcpyAsync(residuals_host_V, d_residualsVDCT,
			cm_gpu->vpw * cm_gpu->vph * sizeof(int16_t), cudaMemcpyDeviceToHost,
			streams[2]);
}

__device__ static void gpu_dequantize_block(float *in_data, float *out_data, int index)
{
	int tid = threadIdx.y * 8 + threadIdx.x;

    uint8_t u = GPU_zigzag_U[tid];
    uint8_t v = GPU_zigzag_UV[tid];

	dct = in_data[tid];

    // Zig-zag and de-quantize
	out_data[v*8+u] = (float) roundf((dct * gpu_quanttbl1D[index][tid]) / GPU_QUANT_VALUE);
}

__shared__ float idct;

__device__ void gpu_dequant_idct_block_8x8(int16_t *in_data, uint8_t *prediction, int w, uint8_t *out_data, int index)
{
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int tid = tid_y * 8 + tid_x;
	int i;

	__shared__ float dct_in[8*8];
	__shared__ float dct_out[8*8];
	__shared__ int16_t tmp[8*8];

	dct_in[tid] = in_data[tid];

	gpu_dequantize_block(dct_in, dct_out, index);

	//this part has to be done in serial, hence synchronizing for every step
	__syncthreads();
	//scale block
	//gpu_scale_block_GPU(dct_out, dct_in);
	a1 = !tid_x ? GPU_ISQRT2 : GPU_SCALE_VALUE;
	a2 = !tid_y ? GPU_ISQRT2 : GPU_SCALE_VALUE;
	dct_in[tid_y*8+tid_x] = dct_out[tid_y*8+tid_x] * a1 * a2;

	__syncthreads();

	// dequantize operation
	idct = 0;
	for(i = 0; i < 8; ++i)
		idct += dct_in[i+8*tid_x] * gpu_dctlookup[tid_y][i];

	dct_out[tid_x*8+tid_y] = idct;

	__syncthreads();
	//transpose block
	dct_in[tid_y*8+tid_x] = dct_out[tid_x*8+tid_y];

	__syncthreads();
	//dequantize operation
	idct = 0;
	for(i = 0; i < 8; ++i)
		idct += dct_in[i+8*tid_x] * gpu_dctlookup[tid_y][i];

	dct_out[tid_x*8+tid_y] = idct;
	__syncthreads();
	//transpose block
	dct_in[tid_y*8+tid_x] = dct_out[tid_x*8+tid_y];
	__syncthreads();
	// Prediction block, cast to legal values for accurate handling
	tmp[tid_y*8+tid_x] = (int16_t)dct_in[tid_y*8+tid_x] + (int16_t)prediction[tid_y*w+tid_x];

	//make sure values are legal
	if (tmp[tid_y*8+tid_x] < 0)
		tmp[tid_y*8+tid_x] = 0;

	else if (tmp[tid_y*8+tid_x] > 255)
		tmp[tid_y*8+tid_x] = 255;

	out_data[tid_y*w+tid_x] = tmp[tid_y*8+tid_x];
}

__global__ void gpu_dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint8_t *out_data, int index)
{
	int bid = blockIdx.y * (width / 8) + blockIdx.x;
	int bid_x = blockIdx.x * 8;
	int bid_y = blockIdx.y * width * 8;
	int offset =  bid_x+bid_y;

	gpu_dequant_idct_block_8x8(in_data+(bid*64), prediction+offset, width, out_data+offset, index);
}

__host__ void idct_test(c63_common_gpu *cm_gpu, uint8_t* d_predictedY, uint8_t* d_predictedU, uint8_t* d_predictedV,
		int16_t *d_residualsYDCT, int16_t *d_residualsUDCT, int16_t *d_residualsVDCT,uint8_t *d_current_reconsY,
		uint8_t *d_current_reconsU, uint8_t *d_current_reconsV, cudaStream_t *streams)
{
	const dim3 threadsPerBlock(8, 8);
	const dim3 numBlocks(cm_gpu->ypw / threadsPerBlock.x, cm_gpu->yph / threadsPerBlock.y);
	const dim3 numBlocks2(cm_gpu->upw / threadsPerBlock.x, cm_gpu->uph / threadsPerBlock.y);
	const dim3 numBlocks3(cm_gpu->vpw / threadsPerBlock.x, cm_gpu->vph / threadsPerBlock.y);


	gpu_dequantize_idct<<<numBlocks, threadsPerBlock, 0, streams[0]>>>
	(d_residualsYDCT, d_predictedY, cm_gpu->ypw, d_current_reconsY, 0);

	gpu_dequantize_idct<<<numBlocks2, threadsPerBlock, 0, streams[1]>>>
	(d_residualsUDCT, d_predictedU, cm_gpu->upw, d_current_reconsU, 1);

	gpu_dequantize_idct<<<numBlocks3, threadsPerBlock, 0, streams[2]>>>
	(d_residualsVDCT, d_predictedV, cm_gpu->vpw, d_current_reconsV, 1);
}








