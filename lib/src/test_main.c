#include "aner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float max_abs(const float *a, const float *b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

static void fill_matrix(float *W, int n, float diag, float off, int stride) {
    memset(W, 0, sizeof(float) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++) {
        W[i * n + i] = diag;
        W[i * n + ((i + stride) % n)] = off;
    }
}

static void fill_state(float *X, int channels, int spatial) {
    for (int c = 0; c < channels; c++)
        for (int s = 0; s < spatial; s++)
            X[c * spatial + s] = 0.02f * (float)(((c + 3 * s) % 9) - 4);
}

static int expect_close(const char *name, const float *got, const float *ref, int n, float tol) {
    float err = max_abs(got, ref, n);
    printf("%s max abs error %.3e\n", name, err);
    if (err > tol) {
        printf("  sample got %.5f ref %.5f\n", got[0], ref[0]);
        return 1;
    }
    return 0;
}

static int test_cpu_residual(void) {
    const int n = 4, sp = 3;
    float W[16], X[12], Y[12], ref[12];
    fill_matrix(W, n, 0.5f, 0.25f, 1);
    fill_state(X, n, sp);
    AnerGraph *g = aner_graph_new();
    int x = aner_input(g, n);
    int y = aner_conv(g, x, n, n, W);
    int z = aner_add(g, y, x);
    int s = aner_sigmoid(g, z);
    aner_output(g, s);
    for (int col = 0; col < sp; col++) {
        float acc[4] = {0};
        for (int r = 0; r < n; r++)
            for (int k = 0; k < n; k++)
                acc[r] += W[r * n + k] * X[k * sp + col];
        for (int r = 0; r < n; r++) {
            float v = acc[r] + X[r * sp + col];
            ref[r * sp + col] = 1.f / (1.f + expf(-v));
        }
    }
    AnerProgram *p = aner_compile(g, sp);
    aner_graph_free(g);
    const float *ins[] = {X};
    float *outs[] = {Y};
    int rc = aner_run(p, ins, outs, NULL);
    int bad = rc != 0 || expect_close("cpu residual sigmoid", Y, ref, n * sp, 1e-5f);
    aner_program_free(p);
    return bad;
}

static int test_ane_product_and_tiles(void) {
    const int n = 512, sp = 64, batch = 128;
    float *W0 = calloc((size_t)n * n, sizeof(float));
    float *W1 = calloc((size_t)n * n, sizeof(float));
    float *X = calloc((size_t)n * batch, sizeof(float));
    float *Y = calloc((size_t)n * batch, sizeof(float));
    float *ref = calloc((size_t)n * batch, sizeof(float));
    fill_matrix(W0, n, 0.5f, 0.125f, 1);
    fill_matrix(W1, n, 0.25f, -0.0625f, 3);
    fill_state(X, n, batch);

    AnerGraph *g = aner_graph_new();
    int x = aner_input(g, n);
    int h = aner_conv(g, x, n, n, W0);
    int y = aner_conv(g, h, n, n, W1);
    aner_output(g, y);

    const float *ins[] = {X};
    float *refOut[] = {ref};
    if (aner_run_reference(g, batch, ins, refOut) != 0)
        return 1;

    AnerProgram *tiled = aner_compile(g, sp);
    int on = aner_uses_neural_engine(tiled);
    double compileMs = aner_compile_milliseconds(tiled);
    double execMs = 0;
    float *outs[] = {Y};
    int rc = aner_run_batch(tiled, ins, outs, batch, &execMs);
    printf("two-matrix chain on %s, compile %.1f ms, batch %d eval %.2f ms, rc %d\n",
           on ? "Neural Engine" : "CPU", compileMs, batch, execMs, rc);
    int bad = !on || rc != 0 || expect_close("ane product vs float32 reference", Y, ref, n * batch, 2e-3f);
    aner_graph_free(g);
    aner_program_free(tiled);
    free(W0); free(W1); free(X); free(Y); free(ref);
    return bad;
}

static int test_ane_residual(void) {
    const int n = 512, sp = 64;
    float *W = calloc((size_t)n * n, sizeof(float));
    float *X = calloc((size_t)n * sp, sizeof(float));
    float *Y = calloc((size_t)n * sp, sizeof(float));
    float *conv = calloc((size_t)n * sp, sizeof(float));
    float *ref = calloc((size_t)n * sp, sizeof(float));
    fill_matrix(W, n, 0.5f, 0.1f, 1);
    fill_state(X, n, sp);

    AnerGraph *convOnly = aner_graph_new();
    int x0 = aner_input(convOnly, n);
    int c0 = aner_conv(convOnly, x0, n, n, W);
    aner_output(convOnly, c0);
    AnerProgram *cp = aner_compile(convOnly, sp);
    const float *ins[] = {X};
    float *convOut[] = {conv};
    aner_run(cp, ins, convOut, NULL);
    for (int i = 0; i < n * sp; i++)
        ref[i] = conv[i] + X[i];

    AnerGraph *g = aner_graph_new();
    int x = aner_input(g, n);
    int c = aner_conv(g, x, n, n, W);
    int z = aner_add(g, c, x);
    aner_output(g, z);
    AnerProgram *p = aner_compile(g, sp);
    float *outs[] = {Y};
    double ms = 0;
    int rc = aner_run(p, ins, outs, &ms);
    int on = aner_uses_neural_engine(p);
    printf("residual conv+add on %s (%.2f ms, rc %d)\n", on ? "Neural Engine" : "CPU", ms, rc);
    float tol = on ? 2e-3f : 1e-5f;
    int bad = rc != 0 || expect_close("residual", Y, ref, n * sp, tol);
    if (!on)
        bad = 1;
    aner_graph_free(convOnly);
    aner_graph_free(g);
    aner_program_free(cp);
    aner_program_free(p);
    free(W); free(X); free(Y); free(conv); free(ref);
    return bad;
}

static int test_ane_sigmoid(void) {
    const int n = 512, sp = 64;
    float *X = calloc((size_t)n * sp, sizeof(float));
    float *Y = calloc((size_t)n * sp, sizeof(float));
    float *ref = calloc((size_t)n * sp, sizeof(float));
    fill_state(X, n, sp);
    AnerGraph *g = aner_graph_new();
    int x = aner_input(g, n);
    int y = aner_sigmoid(g, x);
    aner_output(g, y);
    if (aner_run_reference(g, sp, (const float *const[]){X}, (float *const[]){ref}) != 0)
        return 1;
    AnerProgram *p = aner_compile(g, sp);
    double ms = 0;
    int rc = aner_run(p, (const float *const[]){X}, (float *const[]){Y}, &ms);
    int on = aner_uses_neural_engine(p);
    printf("sigmoid on %s (%.2f ms, rc %d)\n", on ? "Neural Engine" : "CPU", ms, rc);
    int bad = !on || rc != 0 || expect_close("sigmoid", Y, ref, n * sp, 2e-3f);
    aner_graph_free(g);
    aner_program_free(p);
    free(X); free(Y); free(ref);
    return bad;
}

static int test_fields_and_product(void) {
    const int n = 512, sp = 64;
    float *A = calloc((size_t)n * n, sizeof(float));
    float *B = calloc((size_t)n * n, sizeof(float));
    float *X0 = calloc((size_t)n * sp, sizeof(float));
    float *X1 = calloc((size_t)n * sp, sizeof(float));
    float *prod = calloc((size_t)n * sp, sizeof(float));
    float *sum = calloc((size_t)n * sp, sizeof(float));
    float *refProd = calloc((size_t)n * sp, sizeof(float));
    float *refSum = calloc((size_t)n * sp, sizeof(float));
    fill_matrix(A, n, 0.5f, 0.05f, 1);
    fill_matrix(B, n, 0.25f, -0.02f, 2);
    fill_state(X0, n, sp);
    for (int i = 0; i < n * sp; i++)
        X1[i] = 0.03f * (float)(((i * 5) % 7) - 3);

    AnerGraph *g = aner_graph_new();
    int x0 = aner_input(g, n);
    int x1 = aner_input(g, n);
    int y = aner_mul(g, x0, x1);
    int ax = aner_conv(g, x0, n, n, A);
    int bx = aner_conv(g, x1, n, n, B);
    int z = aner_add(g, ax, bx);
    aner_output(g, y);
    aner_output(g, z);
    const float *ins[] = {X0, X1};
    float *refOut[] = {refProd, refSum};
    float *got[] = {prod, sum};
    if (aner_run_reference(g, sp, ins, refOut) != 0)
        return 1;
    AnerProgram *p = aner_compile(g, sp);
    double ms = 0;
    int rc = aner_run(p, ins, got, &ms);
    int on = aner_uses_neural_engine(p);
    printf("two fields, product and linear combination on %s (%.2f ms, rc %d)\n",
           on ? "Neural Engine" : "CPU", ms, rc);
    int bad = !on || rc != 0
        || expect_close("field product", prod, refProd, n * sp, 2e-3f)
        || expect_close("linear combination", sum, refSum, n * sp, 2e-3f);
    aner_graph_free(g);
    aner_program_free(p);
    free(A); free(B); free(X0); free(X1); free(prod); free(sum); free(refProd); free(refSum);
    return bad;
}

static int test_const_scale_series(void) {
    const int n = 512, sp = 64;
    float *bias = malloc(sizeof(float) * (size_t)n);
    float *scale = malloc(sizeof(float) * (size_t)n);
    float *X = calloc((size_t)n * sp, sizeof(float));
    float *Y = calloc((size_t)n * sp, sizeof(float));
    float *ref = calloc((size_t)n * sp, sizeof(float));
    for (int c = 0; c < n; c++) {
        bias[c] = 0.05f * (float)((c % 7) - 3);
        scale[c] = 0.5f + 0.01f * (float)(c % 5);
    }
    fill_state(X, n, sp);
    const float series[] = {0.1f, -0.25f, 0.05f};

    AnerGraph *g = aner_graph_new();
    int x = aner_input(g, n);
    int shifted = aner_add(g, x, aner_const(g, n, bias));
    int scaled = aner_scale(g, shifted, scale);
    int y = aner_horner(g, scaled, series, 3);
    aner_output(g, y);
    const float *ins[] = {X};
    float *refOut[] = {ref};
    float *got[] = {Y};
    if (aner_run_reference(g, sp, ins, refOut) != 0)
        return 1;
    AnerProgram *p = aner_compile(g, sp);
    double ms = 0;
    int rc = aner_run(p, ins, got, &ms);
    int on = aner_uses_neural_engine(p);
    printf("bias, scale, and quadratic series on %s (%.2f ms, rc %d)\n",
           on ? "Neural Engine" : "CPU", ms, rc);
    int bad = !on || rc != 0 || expect_close("series", Y, ref, n * sp, 2e-3f);
    aner_graph_free(g);
    aner_program_free(p);
    free(bias); free(scale); free(X); free(Y); free(ref);
    return bad;
}

static int test_gelu_and_repeat(void) {
    const int n = 512, sp = 64;
    float *A = calloc((size_t)n * n, sizeof(float));
    float *X = calloc((size_t)n * sp, sizeof(float));
    float *G = calloc((size_t)n * sp, sizeof(float));
    float *R = calloc((size_t)n * sp, sizeof(float));
    float *refG = calloc((size_t)n * sp, sizeof(float));
    float *refR = calloc((size_t)n * sp, sizeof(float));
    for (int i = 0; i < n; i++)
        A[i * n + i] = 0.5f;
    fill_state(X, n, sp);

    AnerGraph *g = aner_graph_new();
    int x = aner_input(g, n);
    int gelu = aner_gelu(g, x);
    int stepped = aner_repeat(g, x, 8, A);
    aner_output(g, gelu);
    aner_output(g, stepped);
    const float *ins[] = {X};
    if (aner_run_reference(g, sp, ins, (float *const[]){refG, refR}) != 0)
        return 1;
    AnerProgram *p = aner_compile(g, sp);
    double ms = 0;
    int rc = aner_run(p, ins, (float *const[]){G, R}, &ms);
    int on = aner_uses_neural_engine(p);
    printf("gelu and 8-step map on %s (%.2f ms, rc %d)\n", on ? "Neural Engine" : "CPU", ms, rc);
    int bad = !on || rc != 0
        || expect_close("gelu", G, refG, n * sp, 2e-3f)
        || expect_close("repeat", R, refR, n * sp, 2e-3f);
    aner_graph_free(g);
    aner_program_free(p);
    free(A); free(X); free(G); free(R); free(refG); free(refR);
    return bad;
}

static int test_gpu_chain(void) {
    const int n = 512, sp = 64, batch = 512;
    float *A = calloc((size_t)n * n, sizeof(float));
    float *X = calloc((size_t)n * batch, sizeof(float));
    float *Y = calloc((size_t)n * batch, sizeof(float));
    float *ref = calloc((size_t)n * batch, sizeof(float));
    fill_matrix(A, n, 0.5f, 0.1f, 1);
    fill_state(X, n, batch);

    AnerGraph *g = aner_graph_new();
    aner_output(g, aner_repeat(g, aner_input(g, n), 8, A));
    const float *ins[] = {X};
    if (aner_run_reference(g, batch, ins, (float *const[]){ref}) != 0)
        return 1;

    AnerProgram *p = aner_compile_scheduled(g, sp, ANER_SCHED_GPU);
    int on_gpu = aner_uses_gpu(p);
    int on_ane = aner_uses_neural_engine(p);
    double ms = 0;
    int rc = aner_run_batch(p, ins, (float *const[]){Y}, batch, &ms);
    printf("GPU 8-step chain: on_gpu %d, on_ane %d, batch %d eval %.2f ms, rc %d\n",
           on_gpu, on_ane, batch, ms, rc);
    int bad = !on_gpu || on_ane || rc != 0 || expect_close("gpu chain vs float32 reference", Y, ref, n * batch, 2e-3f);
    aner_graph_free(g);
    aner_program_free(p);
    free(A); free(X); free(Y); free(ref);
    return bad;
}

static int test_split_matches_ane(void) {
    const int n = 512, sp = 64, batch = 4096;
    float *A = calloc((size_t)n * n, sizeof(float));
    float *X = calloc((size_t)n * batch, sizeof(float));
    float *Ysplit = calloc((size_t)n * batch, sizeof(float));
    float *Yane = calloc((size_t)n * batch, sizeof(float));
    fill_matrix(A, n, 0.9f, 0.03f, 1);
    fill_state(X, n, batch);

    AnerGraph *g = aner_graph_new();
    aner_output(g, aner_repeat(g, aner_input(g, n), 64, A));
    const float *ins[] = {X};

    AnerProgram *split = aner_compile_scheduled(g, sp, ANER_SCHED_SPLIT);
    AnerProgram *ane = aner_compile_scheduled(g, sp, ANER_SCHED_ANE);
    int on_ane = aner_uses_neural_engine(split);
    int on_gpu = aner_uses_gpu(split);
    double splitMs = 0, aneMs = 0;
    int rcs = aner_run_batch(split, ins, (float *const[]){Ysplit}, batch, &splitMs);
    int rca = aner_run_batch(ane, ins, (float *const[]){Yane}, batch, &aneMs);
    printf("split: ane %d + gpu %d, batch %d, split eval %.2f ms vs ane-only %.2f ms, rc %d/%d\n",
           on_ane, on_gpu, batch, splitMs, aneMs, rcs, rca);
    /* Both engines compute in fp16, so the split matches the ANE-only run to
       fp16 rounding, not to the float32 interpreter. */
    int bad = !on_ane || !on_gpu || rcs != 0 || rca != 0
        || expect_close("split vs ane-only", Ysplit, Yane, n * batch, 3e-2f);
    aner_graph_free(g);
    aner_program_free(split);
    aner_program_free(ane);
    free(A); free(X); free(Ysplit); free(Yane);
    return bad;
}

int main(void) {
    int failed = 0;
    failed |= test_cpu_residual();
    failed |= test_ane_product_and_tiles();
    failed |= test_ane_residual();
    failed |= test_ane_sigmoid();
    failed |= test_fields_and_product();
    failed |= test_const_scale_series();
    failed |= test_gelu_and_repeat();
    failed |= test_gpu_chain();
    failed |= test_split_matches_ane();
    printf("%s\n", failed ? "FAILED" : "ok");
    return failed ? 1 : 0;
}
