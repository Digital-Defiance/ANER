#define ACCELERATE_NEW_LAPACK

#include "internal.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int aner_cpu_run(const AnerGraph *graph, int spatial, const float *const *inputs, float *const *outputs) {
    if (!graph || spatial <= 0)
        return -1;
    float **buf = calloc((size_t)graph->nops, sizeof(float *));
    if (!buf)
        return -1;
    int rc = 0;
    for (int i = 0; i < graph->nops; i++) {
        const AnerOp *op = &graph->ops[i];
        size_t n = (size_t)op->channels * (size_t)spatial;
        buf[i] = calloc(n, sizeof(float));
        if (!buf[i]) {
            rc = -1;
            break;
        }
        if (op->kind == ANER_INPUT) {
            if (!inputs || !inputs[op->input_index]) {
                rc = -1;
                break;
            }
            memcpy(buf[i], inputs[op->input_index], sizeof(float) * n);
        } else if (op->kind == ANER_CONST) {
            for (int c = 0; c < op->channels; c++)
                for (int s = 0; s < spatial; s++)
                    buf[i][c * spatial + s] = op->weight[c];
        } else if (op->kind == ANER_CONV) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        op->channels, spatial, op->in_channels, 1.f,
                        op->weight, op->in_channels, buf[op->a], spatial, 0.f, buf[i], spatial);
        } else if (op->kind == ANER_ADD) {
            for (size_t k = 0; k < n; k++)
                buf[i][k] = buf[op->a][k] + buf[op->b][k];
        } else if (op->kind == ANER_MUL) {
            for (size_t k = 0; k < n; k++)
                buf[i][k] = buf[op->a][k] * buf[op->b][k];
        } else if (op->kind == ANER_SIGMOID) {
            for (size_t k = 0; k < n; k++) {
                float x = buf[op->a][k];
                buf[i][k] = 1.f / (1.f + expf(-x));
            }
        }
    }
    if (rc == 0) {
        for (int i = 0; i < graph->nops; i++) {
            if (!graph->ops[i].is_output)
                continue;
            int slot = graph->ops[i].output_index;
            if (!outputs || !outputs[slot]) {
                rc = -1;
                break;
            }
            size_t n = (size_t)graph->ops[i].channels * (size_t)spatial;
            memcpy(outputs[slot], buf[i], sizeof(float) * n);
        }
    }
    for (int i = 0; i < graph->nops; i++)
        free(buf[i]);
    free(buf);
    return rc;
}
