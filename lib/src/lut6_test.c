#include "aner_lut6.h"

#include <stdio.h>
#include <string.h>

static int fails;

static void expect(const char *name, int cond) {
    if (cond)
        return;
    fprintf(stderr, "fail %s\n", name);
    fails++;
}

static void broadcast(float *entries, int width, const char *msb_first) {
    int raw = 1 << width;
    memset(entries, 0, sizeof(float) * 64);
    for (int mask = 0; mask < raw; mask++)
        entries[mask] = msb_first[raw - 1 - mask] == '1' ? 1.f : 0.f;
    for (int i = raw; i < 64; i++)
        entries[i] = entries[i % raw];
}

static float reference(const float *entries, const float x[6]) {
    float acc = 0.f;
    for (int k = 0; k < 64; k++) {
        float term = entries[k];
        for (int bit = 0; bit < 6; bit++) {
            float gate = (k & (1 << bit)) ? x[bit] : (1.f - x[bit]);
            term *= gate;
        }
        acc += term;
    }
    return acc;
}

static void check_close(const char *name, float got, float want, float tol) {
    float err = got - want;
    if (err < 0)
        err = -err;
    if (err <= tol)
        return;
    fprintf(stderr, "fail %s got %.6f want %.6f\n", name, got, want);
    fails++;
}

static void test_xor_batch(void) {
    int32_t inputs[6] = {0, 1, -1, -1, -1, -1};
    int32_t outputs[1] = {2};
    int32_t level[1] = {0};
    int32_t counts[1] = {1};
    float entries[64];
    float wires[4 * 3];
    AnerLUT6Program *program;
    broadcast(entries, 2, "0110");
    memset(wires, 0, sizeof(wires));
    wires[1 * 3 + 0] = 1.f;
    wires[2 * 3 + 1] = 1.f;
    wires[3 * 3 + 0] = 1.f;
    wires[3 * 3 + 1] = 1.f;
    program = aner_lut6_compile(3, inputs, outputs, entries, 1, level, counts, 1);
    expect("xor compiles", program != NULL);
    expect("xor on neural engine", aner_lut6_uses_neural_engine(program));
    expect("xor forward", program && aner_lut6_forward(program, wires, 4, NULL) == 0);
    check_close("0 xor 0", wires[2], 0.f, 1e-4f);
    check_close("1 xor 0", wires[1 * 3 + 2], 1.f, 1e-4f);
    check_close("0 xor 1", wires[2 * 3 + 2], 1.f, 1e-4f);
    check_close("1 xor 1", wires[3 * 3 + 2], 0.f, 1e-4f);
    aner_lut6_free(program);
}

static void test_parity64(void) {
    int32_t inputs[6] = {0, 1, 2, 3, 4, 5};
    int32_t outputs[1] = {6};
    int32_t level[1] = {0};
    int32_t counts[1] = {1};
    float entries[64];
    float wires[64 * 7];
    AnerLUT6Program *program;
    for (int mask = 0; mask < 64; mask++)
        entries[mask] = (__builtin_popcount((unsigned)mask) & 1) ? 1.f : 0.f;
    memset(wires, 0, sizeof(wires));
    for (int mask = 0; mask < 64; mask++) {
        for (int bit = 0; bit < 6; bit++)
            wires[mask * 7 + bit] = (mask & (1 << bit)) ? 1.f : 0.f;
    }
    program = aner_lut6_compile(7, inputs, outputs, entries, 1, level, counts, 1);
    expect("parity compiles", program != NULL);
    expect("parity on neural engine", aner_lut6_uses_neural_engine(program));
    expect("parity forward", program && aner_lut6_forward(program, wires, 64, NULL) == 0);
    for (int mask = 0; mask < 64; mask++) {
        float want = (__builtin_popcount((unsigned)mask) & 1) ? 1.f : 0.f;
        check_close("parity", wires[mask * 7 + 6], want, 1e-4f);
    }
    aner_lut6_free(program);
}

static void test_two_levels_and_soft(void) {
    /* wire0 -NOT-> wire1, then AND(wire1, wire2) -> wire3. */
    int32_t inputs[12] = {
        0, -1, -1, -1, -1, -1,
        1, 2, -1, -1, -1, -1
    };
    int32_t outputs[2] = {1, 3};
    int32_t level[2] = {0, 1};
    int32_t counts[2] = {1, 1};
    float entries[128];
    float wires[4];
    float x[6];
    AnerLUT6Program *program;
    broadcast(entries, 1, "01");
    broadcast(entries + 64, 2, "1000");
    wires[0] = 0.f;
    wires[1] = 0.f;
    wires[2] = 1.f;
    wires[3] = 0.f;
    program = aner_lut6_compile(4, inputs, outputs, entries, 2, level, counts, 2);
    expect("chain compiles", program != NULL);
    expect("chain on neural engine", aner_lut6_uses_neural_engine(program));
    expect("chain forward", program && aner_lut6_forward(program, wires, 1, NULL) == 0);
    check_close("not 0", wires[1], 1.f, 1e-4f);
    check_close("and 1 1", wires[3], 1.f, 1e-4f);
    aner_lut6_free(program);

    wires[0] = 0.25f;
    wires[1] = 0.f;
    wires[2] = 0.5f;
    wires[3] = 0.f;
    program = aner_lut6_compile(4, inputs, outputs, entries, 2, level, counts, 2);
    expect("soft forward", program && aner_lut6_forward(program, wires, 1, NULL) == 0);
    memset(x, 0, sizeof(x));
    x[0] = 0.25f;
    check_close("soft not", wires[1], reference(entries, x), 2e-3f);
    memset(x, 0, sizeof(x));
    x[0] = wires[1];
    x[1] = 0.5f;
    check_close("soft and", wires[3], reference(entries + 64, x), 2e-3f);
    aner_lut6_free(program);
}

static void test_second_tile(void) {
    enum { N = 17, WIRES = 18 };
    int32_t inputs[N * 6];
    int32_t outputs[N];
    int32_t level[N];
    int32_t counts[1] = {N};
    float entries[N * 64];
    float wires[WIRES];
    AnerLUT6Program *program;
    memset(inputs, 0, sizeof(inputs));
    for (int lut = 0; lut < N; lut++) {
        for (int pin = 0; pin < 6; pin++)
            inputs[lut * 6 + pin] = pin == 0 ? 0 : -1;
        outputs[lut] = lut + 1;
        level[lut] = lut;
        broadcast(entries + lut * 64, 1, "10");
    }
    memset(wires, 0, sizeof(wires));
    wires[0] = 1.f;
    program = aner_lut6_compile(WIRES, inputs, outputs, entries, N, level, counts, 1);
    expect("tiles compile", program != NULL);
    expect("tiles on neural engine", aner_lut6_uses_neural_engine(program));
    expect("tiles forward", program && aner_lut6_forward(program, wires, 1, NULL) == 0);
    expect("input preserved", wires[0] == 1.f);
    for (int lut = 0; lut < N; lut++)
        check_close("tile copy", wires[lut + 1], 1.f, 1e-4f);
    aner_lut6_free(program);
}

static void test_too_wide(void) {
    expect("513 wires declined", aner_lut6_compile(513, NULL, NULL, NULL, 0, NULL, NULL, 0) == NULL);
}

int main(void) {
    test_xor_batch();
    test_parity64();
    test_two_levels_and_soft();
    test_second_tile();
    test_too_wide();
    if (fails) {
        fprintf(stderr, "%d lut6 checks failed\n", fails);
        return 1;
    }
    printf("lut6 ok\n");
    return 0;
}
