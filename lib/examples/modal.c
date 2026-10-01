#include "aner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

/* 512-mode damped oscillator, stepped as one resident linear map.
   Each column is an initial condition. The sensor row is the fundamental. */

static double ms_since(uint64_t t0) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0)
        mach_timebase_info(&tb);
    return (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
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
        A[r0 * modes + r0] = c;
        A[r0 * modes + r0 + 1] = -s;
        A[(r0 + 1) * modes + r0] = s;
        A[(r0 + 1) * modes + r0 + 1] = c;
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

static float max_abs(const float *a, const float *b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

int main(int argc, char **argv) {
    int scenarios = 4096;
    int horizon = 128;
    int modes = 512;
    int tile = 64;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--scenarios") == 0 && i + 1 < argc)
            scenarios = atoi(argv[++i]);
        else if (strcmp(argv[i], "--horizon") == 0 && i + 1 < argc)
            horizon = atoi(argv[++i]);
        else if (strcmp(argv[i], "--modes") == 0 && i + 1 < argc)
            modes = atoi(argv[++i]);
        else {
            fprintf(stderr, "modal [--scenarios N] [--horizon D] [--modes C]\n");
            return 2;
        }
    }
    if (modes < 2 || (modes % 2) != 0 || modes > 512 || (modes % 64) != 0
        || scenarios < 1 || scenarios > 65536 || horizon < 1) {
        fprintf(stderr, "modes must be even, a multiple of 64, and at most 512\n");
        return 2;
    }

    float *A = calloc((size_t)modes * modes, sizeof(float));
    float *X0 = calloc((size_t)modes * scenarios, sizeof(float));
    float *XT = calloc((size_t)modes * scenarios, sizeof(float));
    float *R = calloc((size_t)modes * modes, sizeof(float));
    float *Y = calloc((size_t)modes * scenarios, sizeof(float));
    if (!A || !X0 || !XT || !R || !Y)
        return 1;
    build_transition(A, modes);
    build_state(X0, modes, scenarios);
    R[0] = 1.f;
    for (int o = 1; o < modes; o++)
        for (int c = 0; c < modes; c++)
            R[o * modes + c] = 0.02f * cosf(0.05f * (float)(o + 1) * (float)(c + 1));

    AnerGraph *step = aner_graph_new();
    int x = aner_input(step, modes);
    int y = aner_repeat(step, x, horizon, A);
    aner_output(step, y);

    AnerGraph *sense = aner_graph_new();
    int s = aner_input(sense, modes);
    int z = aner_conv(sense, s, modes, modes, R);
    aner_output(sense, z);

    AnerProgram *roll = aner_compile(step, tile);
    AnerProgram *sensor = aner_compile(sense, tile);
    if (!roll || !sensor)
        return 1;

    double rollMs = 0;
    uint64_t t0 = mach_absolute_time();
    int rc = aner_run_batch(roll, (const float *const[]){X0}, (float *const[]){XT}, scenarios, &rollMs);
    double rollWall = ms_since(t0);
    double senseMs = 0;
    int src = aner_run_batch(sensor, (const float *const[]){XT}, (float *const[]){Y}, scenarios, &senseMs);

    int probeN = modes * (scenarios < tile ? scenarios : tile);
    float *probeIn = calloc((size_t)probeN, sizeof(float));
    float *probe = calloc((size_t)probeN, sizeof(float));
    for (int c = 0; c < modes; c++)
        memcpy(probeIn + (size_t)c * (scenarios < tile ? scenarios : tile),
               X0 + (size_t)c * scenarios,
               sizeof(float) * (size_t)(scenarios < tile ? scenarios : tile));
    uint64_t p0 = mach_absolute_time();
    int pref = aner_run_reference(step, scenarios < tile ? scenarios : tile,
                                  (const float *const[]){probeIn}, (float *const[]){probe});
    double probeMs = ms_since(p0);
    float err = 0.f;
    if (pref == 0) {
        int width = scenarios < tile ? scenarios : tile;
        for (int c = 0; c < modes; c++) {
            float d = max_abs(XT + (size_t)c * scenarios, probe + (size_t)c * width, width);
            if (d > err)
                err = d;
        }
    }
    float sensorErr = max_abs(Y, XT, scenarios);

    float *sample = malloc(sizeof(float) * (size_t)scenarios);
    memcpy(sample, Y, sizeof(float) * (size_t)scenarios);
    float p05 = percentile(sample, scenarios, 0.05);
    memcpy(sample, Y, sizeof(float) * (size_t)scenarios);
    float p50 = percentile(sample, scenarios, 0.50);
    memcpy(sample, Y, sizeof(float) * (size_t)scenarios);
    float p95 = percentile(sample, scenarios, 0.95);

    int nmodes = modes / 2;
    int hi = nmodes - 1;
    float scale0 = powf(mode_rho(0, nmodes), (float)horizon);
    float scaleHi = powf(mode_rho(hi, nmodes), (float)horizon);
    double amp0 = 0, ampHi = 0, pred0 = 0, predHi = 0;
    for (int s = 0; s < scenarios; s++) {
        float a = scenario_amp(s);
        amp0 += hypotf(XT[s], XT[scenarios + s]);
        ampHi += hypotf(XT[(size_t)(2 * hi) * scenarios + s], XT[(size_t)(2 * hi + 1) * scenarios + s]);
        pred0 += a * scale0;
        predHi += a * scaleHi;
    }
    amp0 /= scenarios;
    ampHi /= scenarios;
    pred0 /= scenarios;
    predHi /= scenarios;

    printf("Modal forecast\n");
    printf("  %d modes, %d scenarios, %d steps, tile %d\n", modes, scenarios, horizon, tile);
    int rollOn = aner_uses_neural_engine(roll);
    int senseOn = aner_uses_neural_engine(sensor);
    printf("  rollout %s, compile %.1f ms, eval %.1f ms, wall %.1f ms, rc %d\n",
           rollOn ? "Neural Engine" : "CPU",
           aner_compile_milliseconds(roll), rollMs, rollWall, rc);
    printf("  float32 probe of first tile: %.1f ms, max abs error %.3e\n", probeMs, err);
    printf("  sensor %s, eval %.1f ms, fundamental error %.3e, rc %d\n",
           senseOn ? "Neural Engine" : "CPU", senseMs, sensorErr, src);
    printf("Fundamental position after %d steps, across %d initial amplitudes\n", horizon, scenarios);
    printf("  p05 %.4f   p50 %.4f   p95 %.4f\n", p05, p50, p95);
    printf("Modal amplitude, mean measured / undamped-step prediction\n");
    printf("  fundamental  measured %.4f  predicted %.4f  ratio %.4f\n", amp0, pred0, pred0 > 0 ? amp0 / pred0 : 0);
    printf("  highest mode measured %.4f  predicted %.4f  ratio %.4f\n", ampHi, predHi, predHi > 0 ? ampHi / predHi : 0);
    printf("  damping retained  fundamental %.0f%%  highest mode %.0f%%\n", 100.0 * scale0, 100.0 * scaleHi);

    aner_graph_free(step);
    aner_graph_free(sense);
    aner_program_free(roll);
    aner_program_free(sensor);
    free(A); free(X0); free(XT); free(R); free(Y); free(sample); free(probeIn); free(probe);
    if (rc != 0 || src != 0 || !rollOn || err > 5e-2f)
        return 4;
    return 0;
}
