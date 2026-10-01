#include "aner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 128 convolutions, 512 channels, spatial 64, 4.295 GFLOP.
   The earlier peak harness on this M4 Max reported 0.332 ms, 12.94 TFLOPS.
   That speed holds when every layer reads one resident weight. Distinct
   weights are a larger file and run slower, which the second line reports. */

static const int kChannels = 512;
static const int kSpatial = 64;
static const int kDepth = 128;
static const int kWarm = 5;
static const int kTimed = 30;
static const double kPriorMs = 0.332;
static const double kPriorTflops = 12.94;

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static int run_chain(const char *label, int distinct) {
    const int n = kChannels;
    const int depth = kDepth;
    float *weights = calloc((size_t)(distinct ? depth : 1) * n * n, sizeof(float));
    float *X = calloc((size_t)n * kSpatial, sizeof(float));
    float *Y = calloc((size_t)n * kSpatial, sizeof(float));
    if (!weights || !X || !Y)
        return 1;
    int matrices = distinct ? depth : 1;
    for (int layer = 0; layer < matrices; layer++) {
        float *W = weights + (size_t)layer * n * n;
        for (int i = 0; i < n; i++) {
            W[i * n + i] = 0.5f;
            W[i * n + ((i + 1 + layer) % n)] = 0.01f;
        }
    }
    for (int i = 0; i < n * kSpatial; i++)
        X[i] = (i % n == 0) ? 1.f : 0.f;

    AnerGraph *graph = aner_graph_new();
    int cur = aner_input(graph, n);
    for (int layer = 0; layer < depth; layer++) {
        float *W = weights + (size_t)(distinct ? layer : 0) * n * n;
        cur = aner_conv(graph, cur, n, n, W);
        if (cur < 0) {
            fprintf(stderr, "%s conv %d rejected\n", label, layer);
            return 1;
        }
    }
    aner_output(graph, cur);
    AnerProgram *program = aner_compile(graph, kSpatial);
    if (!program) {
        fprintf(stderr, "%s compile failed\n", label);
        return 1;
    }
    int on = aner_uses_neural_engine(program);
    double compileMs = aner_compile_milliseconds(program);
    const float *inputs[] = {X};
    float *outputs[] = {Y};
    for (int i = 0; i < kWarm; i++) {
        if (aner_run(program, inputs, outputs, NULL) != 0) {
            fprintf(stderr, "%s warmup failed\n", label);
            return 1;
        }
    }
    double samples[64];
    double sum = 0;
    for (int i = 0; i < kTimed; i++) {
        double ms = 0;
        if (aner_run(program, inputs, outputs, &ms) != 0) {
            fprintf(stderr, "%s eval failed\n", label);
            return 1;
        }
        samples[i] = ms;
        sum += ms;
    }
    qsort(samples, kTimed, sizeof(double), cmp_double);
    double median = 0.5 * (samples[kTimed / 2 - 1] + samples[kTimed / 2]);
    double gf = 2.0 * n * n * (double)kSpatial * depth / 1e9;
    double tflops = median > 0 ? gf / median : 0;
    printf("%s %d x %dch sp%d on %s\n", label, depth, n, kSpatial, on ? "Neural Engine" : "CPU");
    printf("  compile %.1f ms, median %.3f ms, mean %.3f ms, %.2f TFLOPS\n",
           compileMs, median, sum / kTimed, tflops);
    printf("  prior peak %.3f ms, %.2f TFLOPS, ratio %.2f, y0 %.5f\n",
           kPriorMs, kPriorTflops, tflops / kPriorTflops, Y[0]);
    int bad = !on || (distinct ? tflops < 1.0 : tflops < 10.0);
    aner_graph_free(graph);
    aner_program_free(program);
    free(weights);
    free(X);
    free(Y);
    return bad;
}

int main(void) {
    int bad = 0;
    bad |= run_chain("repeated weight", 0);
    bad |= run_chain("distinct weights", 1);
    printf("%s\n", bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
