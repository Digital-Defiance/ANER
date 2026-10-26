#ifndef ANER_INTERNAL_H
#define ANER_INTERNAL_H

#include "aner.h"

typedef enum {
    ANER_INPUT = 0,
    ANER_CONST,
    ANER_CONV,
    ANER_ADD,
    ANER_MUL,
    ANER_SIGMOID
} AnerOpKind;

typedef struct {
    AnerOpKind kind;
    int a;
    int b;
    int channels;
    int in_channels;
    float *weight;
    int is_output;
    int input_index;
    int output_index;
} AnerOp;

struct AnerGraph {
    AnerOp *ops;
    int nops;
    int cap;
    int ninputs;
    int noutputs;
};

AnerGraph *aner_graph_clone(const AnerGraph *graph);

int aner_cpu_run(const AnerGraph *graph, int spatial, const float *const *inputs, float *const *outputs);

void *aner_ane_compile(const AnerGraph *graph, int spatial, double *compile_ms);
int aner_ane_eval(void *handle, const float *const *inputs, float *const *outputs, double *exec_ms);
void aner_ane_release(void *handle);

/* GPU backend. Compiles the whole graph to an MPSGraph executable at
   optimization level 0 so the work stays on Metal rather than being placed on
   the Neural Engine. A spatial width is fixed at compile time, as on the ANE.
   Returns NULL if Metal is unavailable or the graph cannot be lowered. */
void *aner_gpu_compile(const AnerGraph *graph, int spatial, double *compile_ms);
int aner_gpu_eval(void *handle, const float *const *inputs, float *const *outputs, double *exec_ms);
void aner_gpu_release(void *handle);

#endif
