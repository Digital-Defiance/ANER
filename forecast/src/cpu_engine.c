#define ACCELERATE_NEW_LAPACK

#include "engines.h"

#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

int cpu_chain(const float *A, int channels, int depth, const float *X, float *Y, int spatial) {
    size_t n = (size_t)channels * (size_t)spatial;
    float *buf0 = malloc(sizeof(float) * n);
    float *buf1 = malloc(sizeof(float) * n);
    if (!buf0 || !buf1) {
        free(buf0);
        free(buf1);
        return -1;
    }
    memcpy(buf0, X, sizeof(float) * n);
    float *in = buf0;
    float *out = buf1;
    for (int layer = 0; layer < depth; layer++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    channels, spatial, channels, 1.f,
                    A, channels, in, spatial, 0.f, out, spatial);
        float *swap = in;
        in = out;
        out = swap;
    }
    memcpy(Y, in, sizeof(float) * n);
    free(buf0);
    free(buf1);
    return 0;
}
