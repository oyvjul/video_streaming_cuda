#ifndef C63
#define C63

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_FILELENGTH 200
#define DEFAULT_OUTPUT_FILE "a.mjpg"

#define COLOR_COMPONENTS 3

#define Y_COMPONENT 0
#define U_COMPONENT 1
#define V_COMPONENT 2

#define YX 2
#define YY 2
#define UX 1
#define UY 1
#define VX 1
#define VY 1

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

struct yuv
{
  uint8_t *Y;
  uint8_t *U;
  uint8_t *V;
};

struct dct
{
  int16_t *Ydct;
  int16_t *Udct;
  int16_t *Vdct;
};

typedef struct yuv yuv_t;
typedef struct dct dct_t;

struct entropy_ctx
{
  FILE *fp;
  unsigned int bit_buffer;
  unsigned int bit_buffer_width;
};

struct macroblock
{
  int use_mv;
  int8_t mv_x, mv_y;
};

struct frame
{
  yuv_t *orig;        // Original input image
  yuv_t *recons;      // Reconstructed image
  yuv_t *predicted;   // Predicted frame from intra-prediction

  dct_t *residuals;   // Difference between original image and predicted frame

  struct macroblock *mbs[3];
  int keyframe;
};

struct c63_common_cpu
{
  int width, height;
  int ypw, yph, upw, uph, vpw, vph;

  int padw[3], padh[3];

  int mb_cols, mb_rows;

  uint8_t qp;                         // Quality parameter

  int me_search_range;

  uint8_t quanttbl[3][64];

  yuv_t *orig;
  yuv_t *ref_recons;
  dct_t *residuals;
  struct macroblock *mbs[3];

  struct frame *refframe;
  struct frame *curframe;

  int keyframe;

  int framenum;

  int keyframe_interval;
  int frames_since_keyframe;

  struct entropy_ctx e_ctx;
};

struct c63_common_gpu
{
  int width, height;
  int ypw, yph, upw, uph, vpw, vph;

  int padw[3], padh[3];

  int mb_cols, mb_rows;

  uint8_t qp;                         // Quality parameter

  int me_search_range;

  yuv_t *orig;        // Original input image
  yuv_t *recons;      // Reconstructed image
  yuv_t *predicted;   // Predicted frame from intra-prediction

  dct_t *ref_recons;
  dct_t *residuals;   // Difference between original image and predicted frame

  uint8_t quanttbl[3][64];

  int framenum;

  int keyframe_interval;
  int frames_since_keyframe;

  struct entropy_ctx e_ctx;
};

void destroy_frame(struct frame *f);
void dump_image(yuv_t *image, int w, int h, FILE *fp);
void c63_motion_compensate(struct c63_common_cpu *cm);

#endif
