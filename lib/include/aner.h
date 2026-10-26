#ifndef ANER_H
#define ANER_H

#define ANER_VERSION_MAJOR 0
#define ANER_VERSION_MINOR 1
#define ANER_VERSION_PATCH 0
#define ANER_VERSION_STRING "0.1.0"

/* Apple Neural Engine Runtime.
   Static graphs for ensemble numerical work.

   A value is a [channels, spatial] fp32 matrix, row-major. Each column is an
   independent sample: an ensemble member, a scenario, or a batch element.
   Channels are the state. Spatial is chosen at compile time and stays fixed.
   64 is the width that keeps a long chain resident on an M4 Max.

   Coefficients are constants. Linear maps are row-major matrices. aner_repeat
   applies one map for many steps and keeps that matrix resident. aner_const
   writes one value per channel into every column. aner_horner evaluates an
   elementwise power series. aner_gelu is x * sigmoid(1.702 * x).

   The compiler targets the Neural Engine when every channel count is a
   multiple of 64 and at most 512. Otherwise, and if lowering fails, runs use
   the float32 interpreter. aner_run_reference always uses that interpreter. */

typedef struct AnerGraph AnerGraph;
typedef struct AnerProgram AnerProgram;

const char *aner_version(void);

AnerGraph *aner_graph_new(void);
void aner_graph_free(AnerGraph *graph);

/* Returns a tensor id, or -1. */
int aner_input(AnerGraph *graph, int channels);

/* per_channel has one value per channel. Every column receives that value. */
int aner_const(AnerGraph *graph, int channels, const float *per_channel);

/* Row-major matrix, shape out_channels × in_channels. */
int aner_conv(AnerGraph *graph, int input, int out_channels, int in_channels, const float *row_major);

/* Diagonal map. per_channel has one scale per channel of input. */
int aner_scale(AnerGraph *graph, int input, const float *per_channel);

/* Apply one square map, steps times. steps >= 1. The map is channels × channels. */
int aner_repeat(AnerGraph *graph, int input, int steps, const float *square_row_major);

int aner_add(AnerGraph *graph, int a, int b);
int aner_mul(AnerGraph *graph, int a, int b);
int aner_sigmoid(AnerGraph *graph, int input);

/* coeffs_low_to_high[k] is the coefficient of x^k. The series is elementwise. */
int aner_horner(AnerGraph *graph, int input, const float *coeffs_low_to_high, int ncoeff);

/* x * sigmoid(1.702 * x), the Hendrycks–Gimpel approximation. */
int aner_gelu(AnerGraph *graph, int input);

/* Outputs are returned in the order they are marked. */
int aner_output(AnerGraph *graph, int tensor);

/* Copies the graph. The caller's graph may be freed after this returns.
   aner_compile keeps the historical behaviour: the Neural Engine when the
   shape allows, otherwise the float32 interpreter. */
AnerProgram *aner_compile(const AnerGraph *graph, int spatial);

/* Batch scheduling across compute engines, chosen at compile time.

   ANER_SCHED_ANE    Neural Engine only, then the interpreter. The default, and
                     what aner_compile selects.
   ANER_SCHED_GPU    Metal only, then the interpreter.
   ANER_SCHED_SPLIT  Neural Engine and GPU at once. aner_run_batch hands each
                     engine the next tile as it falls idle, so a wide batch runs
                     on both. A single tile (aner_run, or a batch equal to the
                     compiled spatial width) still runs on one engine. If either
                     backend is unavailable the other carries the whole batch;
                     if neither is, the interpreter does.

   Results are fp16 on both engines, so a split batch matches an ANE-only batch
   to fp16 rounding, not to the float32 interpreter. */
typedef enum {
    ANER_SCHED_ANE = 0,
    ANER_SCHED_GPU = 1,
    ANER_SCHED_SPLIT = 2
} AnerSchedule;

AnerProgram *aner_compile_scheduled(const AnerGraph *graph, int spatial, AnerSchedule schedule);

void aner_program_free(AnerProgram *program);
int aner_uses_neural_engine(const AnerProgram *program);
int aner_uses_gpu(const AnerProgram *program);
double aner_compile_milliseconds(const AnerProgram *program);

/* Each input is channels × spatial. Each output buffer is likewise sized. */
int aner_run(AnerProgram *program, const float *const *inputs, float *const *outputs, double *exec_ms);

/* Float32 interpreter. Independent of which device the compiled program uses. */
int aner_run_reference(const AnerGraph *graph, int spatial, const float *const *inputs, float *const *outputs);

/* inputs[i] is channels × batch, row-major. The batch is cut into tiles of
   the compiled spatial length. Pointwise and 1×1 ops treat columns independently,
   so a short final tile is padded. */
int aner_run_batch(AnerProgram *program, const float *const *inputs, float *const *outputs,
                   int batch, double *exec_ms);

#endif
