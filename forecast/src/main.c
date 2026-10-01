#include "engines.h"
#include "router.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

static double ms_since(uint64_t t0) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0)
        mach_timebase_info(&tb);
    return (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

static int self_test(void) {
    int misses = router_audit();
    printf("router audit mismatches %d\n", misses);
    if (misses != 0)
        return 1;

    const int ch = 2, depth = 8;
    float A[4] = {0.9f, -0.1f, 0.1f, 0.9f};
    float x[2] = {1.f, 0.f};
    float y[2], yRef[2] = {1.f, 0.f};
    float cur[2] = {1.f, 0.f};
    for (int step = 0; step < depth; step++) {
        yRef[0] = A[0] * cur[0] + A[1] * cur[1];
        yRef[1] = A[2] * cur[0] + A[3] * cur[1];
        cur[0] = yRef[0];
        cur[1] = yRef[1];
    }
    if (cpu_chain(A, ch, depth, x, y, 1) != 0)
        return 2;
    float cpuErr = max_abs_diff(y, yRef, 2);
    printf("cpu 2x2 chain max abs error %.3e\n", cpuErr);
    if (cpuErr > 1e-5f)
        return 2;

    const int gch = 32, gsp = 16;
    float *gA = calloc((size_t)gch * gch, sizeof(float));
    float *gX = calloc((size_t)gch * gsp, sizeof(float));
    float *gY = calloc((size_t)gch * gsp, sizeof(float));
    float *gRef = calloc((size_t)gch * gsp, sizeof(float));
    for (int i = 0; i < gch; i++) {
        gA[i * gch + i] = 0.5f;
        gA[i * gch + ((i + 1) % gch)] = 0.25f;
    }
    for (int i = 0; i < gch * gsp; i++)
        gX[i] = 0.01f * (float)((i % 9) - 4);
    cpu_chain(gA, gch, 1, gX, gRef, gsp);
    double gpuMs = 0;
    int gpuRc = gpu_chain(gA, gch, 1, gX, gY, gsp, &gpuMs);
    float gpuErr = gpuRc == 0 ? max_abs_diff(gY, gRef, gch * gsp) : -1.f;
    printf("gpu gemm 32x16 max abs error %.3e (rc %d, %.3f ms)\n", gpuErr, gpuRc, gpuMs);

    const int ach = 512, asp = 64, adepth = 2;
    float *aA = calloc((size_t)ach * ach, sizeof(float));
    float *aX = calloc((size_t)ach * asp, sizeof(float));
    float *aY = calloc((size_t)ach * asp, sizeof(float));
    float *aRef = calloc((size_t)ach * asp, sizeof(float));
    for (int i = 0; i < ach; i++)
        aA[i * ach + i] = 0.5f;
    aA[1] = 0.25f;
    for (int s = 0; s < asp; s++) {
        aX[0 * asp + s] = 1.f;
        aX[1 * asp + s] = 0.5f;
    }
    cpu_chain(aA, ach, adepth, aX, aRef, asp);
    double compileMs = 0, execMs = 0;
    int aneRc = ane_chain(aA, ach, adepth, aX, aY, asp, &compileMs, &execMs);
    float aneErr = aneRc == 0 ? max_abs_diff(aY, aRef, ach * asp) : -1.f;
    printf("ane 512x64 depth2 max abs error %.3e (rc %d, compile %.1f ms, eval %.3f ms)\n",
           aneErr, aneRc, compileMs, execMs);
    if (aneRc == 0) {
        printf("  sample y0 %.5f ref %.5f  y1 %.5f ref %.5f\n",
               aY[0], aRef[0], aY[asp], aRef[asp]);
    }

    int failed = gpuRc != 0 || gpuErr > 2e-3f || aneRc != 0 || aneErr > 2e-3f;
    free(gA); free(gX); free(gY); free(gRef);
    free(aA); free(aX); free(aY); free(aRef);
    ane_shutdown();
    return failed ? 3 : 0;
}

static void build_transition(float *A, int modes) {
    int nmodes = modes / 2;
    memset(A, 0, sizeof(float) * (size_t)modes * (size_t)modes);
    for (int i = 0; i < nmodes; i++) {
        double t = (double)i / (double)(nmodes - 1);
        double omega = 0.08 * (1.0 + t);
        double zeta = 0.01 + 0.08 * t;
        double wd = omega * sqrt(1.0 - zeta * zeta);
        double rho = exp(-zeta * omega);
        float c = (float)(rho * cos(wd));
        float s = (float)(rho * sin(wd));
        int r0 = 2 * i;
        int r1 = r0 + 1;
        A[r0 * modes + r0] = c;
        A[r0 * modes + r1] = -s;
        A[r1 * modes + r0] = s;
        A[r1 * modes + r1] = c;
    }
}

static float mode_rho(int mode, int nmodes) {
    double t = (double)mode / (double)(nmodes - 1);
    double omega = 0.08 * (1.0 + t);
    double zeta = 0.01 + 0.08 * t;
    return (float)exp(-zeta * omega);
}

static float scenario_amp(int scenario) {
    return 0.2f + 1.3f * (float)((scenario * 13 + 7) % 100) / 100.f;
}

static void build_state(float *X, int modes, int scenarios) {
    int nmodes = modes / 2;
    for (int s = 0; s < scenarios; s++) {
        float amp = scenario_amp(s);
        for (int i = 0; i < nmodes; i++) {
            float phase = 0.2f * (float)((s + 3 * i) % 13);
            X[(2 * i) * scenarios + s] = amp * cosf(phase);
            X[(2 * i + 1) * scenarios + s] = amp * sinf(phase);
        }
    }
}

static void pack_tile(const float *X, float *tile, int channels, int scenarios, int scenario0, int spatial) {
    for (int c = 0; c < channels; c++)
        memcpy(tile + (size_t)c * spatial,
               X + (size_t)c * scenarios + scenario0,
               sizeof(float) * (size_t)spatial);
}

static void unpack_tile(float *Y, const float *tile, int channels, int scenarios, int scenario0, int spatial) {
    for (int c = 0; c < channels; c++)
        memcpy(Y + (size_t)c * scenarios + scenario0,
               tile + (size_t)c * spatial,
               sizeof(float) * (size_t)spatial);
}

static int cmp_float(const void *a, const void *b) {
    float da = *(const float *)a, db = *(const float *)b;
    return (da > db) - (da < db);
}

static float percentile(float *v, int n, double p) {
    qsort(v, (size_t)n, sizeof(float), cmp_float);
    double idx = p * (double)(n - 1);
    int i = (int)idx;
    if (i >= n - 1)
        return v[n - 1];
    float f = (float)(idx - i);
    return v[i] * (1.f - f) + v[i + 1] * f;
}

static int apply_tile(Engine engine, const float *A, int channels, int depth,
                      const float *X, float *Y, int spatial,
                      double *compile_ms, double *exec_ms, Engine *used) {
    int rc = -1;
    *used = engine;
    *compile_ms = 0;
    *exec_ms = 0;
    if (engine == ENGINE_ANE)
        rc = ane_chain(A, channels, depth, X, Y, spatial, compile_ms, exec_ms);
    if (rc != 0 && engine != ENGINE_CPU) {
        double ms = 0;
        rc = gpu_chain(A, channels, depth, X, Y, spatial, &ms);
        if (rc == 0) {
            *used = ENGINE_GPU;
            *exec_ms = ms;
        }
    }
    if (rc != 0) {
        uint64_t t0 = mach_absolute_time();
        rc = cpu_chain(A, channels, depth, X, Y, spatial);
        *used = ENGINE_CPU;
        *exec_ms = ms_since(t0);
    }
    return rc;
}

static void usage(void) {
    fprintf(stderr,
            "modal-forecast [--scenarios N] [--horizon D] [--modes C] [--self-test]\n"
            "  Default: 512-mode damped system, 4096 initial conditions, 128 steps.\n"
            "  Rollout tiles of 64 go to the Neural Engine. The 512-sensor\n"
            "  projection of a wide ensemble goes to the GPU. A float32 probe\n"
            "  and the response percentiles stay on the CPU.\n");
}

int main(int argc, char **argv) {
    int scenarios = 4096;
    int horizon = 128;
    int modes = 512;
    int do_test = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--self-test") == 0)
            do_test = 1;
        else if (strcmp(argv[i], "--scenarios") == 0 && i + 1 < argc)
            scenarios = atoi(argv[++i]);
        else if (strcmp(argv[i], "--horizon") == 0 && i + 1 < argc)
            horizon = atoi(argv[++i]);
        else if (strcmp(argv[i], "--modes") == 0 && i + 1 < argc)
            modes = atoi(argv[++i]);
        else {
            usage();
            return 2;
        }
    }
    if (do_test)
        return self_test();
    if (modes < 2 || (modes % 2) != 0 || modes > 512 || scenarios < 1 || horizon < 1 || scenarios > 65536) {
        usage();
        return 2;
    }

    printf("Modal forecast\n");
    printf("  %d modes, %d scenarios, %d steps\n", modes, scenarios, horizon);

    float *A = calloc((size_t)modes * modes, sizeof(float));
    float *X0 = calloc((size_t)modes * scenarios, sizeof(float));
    float *XT = calloc((size_t)modes * scenarios, sizeof(float));
    if (!A || !X0 || !XT)
        return 1;
    build_transition(A, modes);
    build_state(X0, modes, scenarios);

    Tile tiles[1024];
    int nt = plan_rollout(modes, scenarios, horizon, 0, tiles, 1024);
    if (nt < 0)
        return 1;
    Engine sensorEngine = route_engine(1, modes, scenarios, 0);
    printf("  rollout tiles %d, first tile %s spatial %d\n", nt, engine_name(tiles[0].engine), tiles[0].spatial);
    printf("  sensor bank (%d x %d) -> %s\n", modes, scenarios, engine_name(sensorEngine));

    int tileCap = 0;
    for (int i = 0; i < nt; i++)
        if (tiles[i].spatial > tileCap)
            tileCap = tiles[i].spatial;
    float *tileIn = calloc((size_t)modes * tileCap, sizeof(float));
    float *tileOut = calloc((size_t)modes * tileCap, sizeof(float));
    double compileTotal = 0, rolloutMs = 0;
    int usedCount[3] = {0, 0, 0};
    uint64_t roll0 = mach_absolute_time();
    for (int i = 0; i < nt; i++) {
        pack_tile(X0, tileIn, modes, scenarios, tiles[i].scenario0, tiles[i].spatial);
        double compileMs = 0, execMs = 0;
        Engine used = tiles[i].engine;
        int rc = apply_tile(used, A, modes, horizon, tileIn, tileOut, tiles[i].spatial,
                            &compileMs, &execMs, &used);
        if (rc != 0) {
            fprintf(stderr, "tile %d failed\n", i);
            return 1;
        }
        unpack_tile(XT, tileOut, modes, scenarios, tiles[i].scenario0, tiles[i].spatial);
        compileTotal += compileMs;
        rolloutMs += execMs;
        usedCount[used]++;
    }
    double rollWall = ms_since(roll0);
    printf("  rollout wall %.1f ms, eval %.1f ms, compile %.1f ms\n", rollWall, rolloutMs, compileTotal);
    printf("  tiles executed  CPU %d  GPU %d  Neural Engine %d\n",
           usedCount[ENGINE_CPU], usedCount[ENGINE_GPU], usedCount[ENGINE_ANE]);

    float *probe = calloc((size_t)modes * tiles[0].spatial, sizeof(float));
    float *probeIn = calloc((size_t)modes * tiles[0].spatial, sizeof(float));
    pack_tile(X0, probeIn, modes, scenarios, 0, tiles[0].spatial);
    uint64_t probe0 = mach_absolute_time();
    cpu_chain(A, modes, horizon, probeIn, probe, tiles[0].spatial);
    double probeMs = ms_since(probe0);
    float *got = calloc((size_t)modes * tiles[0].spatial, sizeof(float));
    pack_tile(XT, got, modes, scenarios, 0, tiles[0].spatial);
    float probeErr = max_abs_diff(got, probe, modes * tiles[0].spatial);
    printf("  cpu fp32 probe of tile 0: %.1f ms, max abs error %.3e\n", probeMs, probeErr);

    float *R = calloc((size_t)modes * modes, sizeof(float));
    float *Y = calloc((size_t)modes * scenarios, sizeof(float));
    R[0] = 1.f;
    for (int o = 1; o < modes; o++)
        for (int c = 0; c < modes; c++)
            R[o * modes + c] = 0.02f * cosf(0.05f * (float)(o + 1) * (float)(c + 1));
    double sensorMs = 0;
    uint64_t sensor0 = mach_absolute_time();
    int sensorRc = -1;
    if (sensorEngine == ENGINE_GPU)
        sensorRc = gpu_chain(R, modes, 1, XT, Y, scenarios, &sensorMs);
    if (sensorRc != 0) {
        sensorRc = cpu_chain(R, modes, 1, XT, Y, scenarios);
        sensorMs = ms_since(sensor0);
        sensorEngine = ENGINE_CPU;
    }
    float sensorErr = max_abs_diff(Y, XT, scenarios);
    printf("  sensor 0 vs fundamental state, max abs error %.3e on %s (%.1f ms)\n",
           sensorErr, engine_name(sensorEngine), sensorMs);

    float *sample = malloc(sizeof(float) * (size_t)scenarios);
    memcpy(sample, Y, sizeof(float) * (size_t)scenarios);
    float p05 = percentile(sample, scenarios, 0.05);
    memcpy(sample, Y, sizeof(float) * (size_t)scenarios);
    float p50 = percentile(sample, scenarios, 0.50);
    memcpy(sample, Y, sizeof(float) * (size_t)scenarios);
    float p95 = percentile(sample, scenarios, 0.95);

    double amp0 = 0, ampHi = 0, pred0 = 0, predHi = 0;
    int hi = modes / 2 - 1;
    float rho0 = mode_rho(0, modes / 2);
    float rhoHi = mode_rho(hi, modes / 2);
    float scale0 = powf(rho0, (float)horizon);
    float scaleHi = powf(rhoHi, (float)horizon);
    for (int s = 0; s < scenarios; s++) {
        float a0 = hypotf(XT[0 * scenarios + s], XT[1 * scenarios + s]);
        float aH = hypotf(XT[(2 * hi) * scenarios + s], XT[(2 * hi + 1) * scenarios + s]);
        float a = scenario_amp(s);
        amp0 += a0;
        ampHi += aH;
        pred0 += a * scale0;
        predHi += a * scaleHi;
    }
    amp0 /= scenarios;
    ampHi /= scenarios;
    pred0 /= scenarios;
    predHi /= scenarios;

    printf("\nFundamental position after %d steps, across %d initial amplitudes\n", horizon, scenarios);
    printf("  p05 %.4f   p50 %.4f   p95 %.4f\n", p05, p50, p95);
    printf("Modal amplitude, mean measured / undamped-step prediction\n");
    printf("  fundamental  measured %.4f  predicted %.4f  ratio %.4f\n", amp0, pred0, pred0 > 0 ? amp0 / pred0 : 0);
    printf("  highest mode measured %.4f  predicted %.4f  ratio %.4f\n", ampHi, predHi, predHi > 0 ? ampHi / predHi : 0);
    printf("  damping retained  fundamental %.0f%%  highest mode %.0f%%\n",
           100.0 * scale0, 100.0 * scaleHi);

    free(A); free(X0); free(XT); free(tileIn); free(tileOut);
    free(probe); free(probeIn); free(got); free(R); free(Y); free(sample);
    ane_shutdown();
    return probeErr > 5e-2f ? 4 : 0;
}
