#include "aner_lut6.h"

#include "aner.h"

#include <stdlib.h>
#include <string.h>

/* Sixteen LUTs per tile. Stage 0 is 16 * 32 channels. Later stages pad the
   pair layout so every tensor width is a multiple of 64. */
#define TILE 16
#define XSTRIDE 8
#define XCH (TILE * XSTRIDE)

struct AnerLUT6Program {
    AnerProgram *program;
    int wires;
    double compile_ms;
};

static int state_channels(int wires) {
    int channels;
    if (wires <= 0 || wires > 512)
        return 0;
    channels = (wires + 63) & ~63;
    return channels <= 512 ? channels : 0;
}

static float *zeros(size_t n) {
    return calloc(n, sizeof(float));
}

static void put(float *matrix, int cols, int row, int col, float value) {
    matrix[(size_t)row * (size_t)cols + (size_t)col] = value;
}

static void fill_broadcast(float *matrix, int out_stride, int pairs, int bit) {
    for (int lut = 0; lut < TILE; lut++) {
        for (int s = 0; s < pairs; s++) {
            put(matrix, XCH, lut * out_stride + s, lut * XSTRIDE + bit, 1.f);
        }
    }
}

static void fill_pairs(float *even, float *diff, int out_ch, int in_ch, int out_stride, int in_stride, int pairs) {
    for (int lut = 0; lut < TILE; lut++) {
        for (int s = 0; s < pairs; s++) {
            int row = lut * out_stride + s;
            int lo = lut * in_stride + 2 * s;
            put(even, in_ch, row, lo, 1.f);
            put(diff, in_ch, row, lo, -1.f);
            put(diff, in_ch, row, lo + 1, 1.f);
        }
    }
    (void)out_ch;
}

typedef struct {
    float *x0;
    float *even[5];
    float *diff[5];
    float *xb[5];
} Geom;

static const int k_out_ch[5] = {256, 128, 64, 64, 64};
static const int k_in_ch[5] = {512, 256, 128, 64, 64};
static const int k_out_stride[5] = {16, 8, 4, 4, 4};
static const int k_in_stride[5] = {32, 16, 8, 4, 4};
static const int k_pairs[5] = {16, 8, 4, 2, 1};
static const int k_bit[5] = {1, 2, 3, 4, 5};

static int geom_build(Geom *geom) {
    memset(geom, 0, sizeof(*geom));
    geom->x0 = zeros((size_t)512 * XCH);
    if (!geom->x0)
        return -1;
    fill_broadcast(geom->x0, 32, 32, 0);
    for (int stage = 0; stage < 5; stage++) {
        size_t pair_n = (size_t)k_out_ch[stage] * (size_t)k_in_ch[stage];
        size_t x_n = (size_t)k_out_ch[stage] * XCH;
        geom->even[stage] = zeros(pair_n);
        geom->diff[stage] = zeros(pair_n);
        geom->xb[stage] = zeros(x_n);
        if (!geom->even[stage] || !geom->diff[stage] || !geom->xb[stage])
            return -1;
        fill_pairs(geom->even[stage], geom->diff[stage], k_out_ch[stage], k_in_ch[stage],
                   k_out_stride[stage], k_in_stride[stage], k_pairs[stage]);
        fill_broadcast(geom->xb[stage], k_out_stride[stage], k_pairs[stage], k_bit[stage]);
    }
    return 0;
}

static void geom_free(Geom *geom) {
    free(geom->x0);
    for (int stage = 0; stage < 5; stage++) {
        free(geom->even[stage]);
        free(geom->diff[stage]);
        free(geom->xb[stage]);
    }
}

static int lerp_bits(AnerGraph *graph, int value, int extracted, const Geom *geom) {
    for (int stage = 0; stage < 5; stage++) {
        int even = aner_conv(graph, value, k_out_ch[stage], k_in_ch[stage], geom->even[stage]);
        int diff = aner_conv(graph, value, k_out_ch[stage], k_in_ch[stage], geom->diff[stage]);
        int gate = aner_conv(graph, extracted, k_out_ch[stage], XCH, geom->xb[stage]);
        int prod, sum;
        if (even < 0 || diff < 0 || gate < 0)
            return -1;
        prod = aner_mul(graph, diff, gate);
        if (prod < 0)
            return -1;
        sum = aner_add(graph, even, prod);
        if (sum < 0)
            return -1;
        value = sum;
    }
    return value;
}

static int eval_tile(AnerGraph *graph, int state, const Geom *geom, int channels,
                     const int32_t *inputs, const int32_t *outputs, const float *entries,
                     const int32_t *lut_index, int count) {
    float *extract = zeros((size_t)XCH * (size_t)channels);
    float *low = zeros(512);
    float *delta = zeros(512);
    float *place = zeros((size_t)channels * 64);
    int extracted, spread, base, step, placed, result;
    if (!extract || !low || !delta || !place) {
        free(extract);
        free(low);
        free(delta);
        free(place);
        return -1;
    }
    for (int slot = 0; slot < count && slot < TILE; slot++) {
        int lut = lut_index[slot];
        const int32_t *pins = inputs + (size_t)lut * 6;
        const float *table = entries + (size_t)lut * 64;
        for (int pin = 0; pin < 6; pin++) {
            if (pins[pin] >= 0)
                put(extract, channels, slot * XSTRIDE + pin, pins[pin], 1.f);
        }
        for (int s = 0; s < 32; s++) {
            float a = table[2 * s];
            float b = table[2 * s + 1];
            low[slot * 32 + s] = a;
            delta[slot * 32 + s] = b - a;
        }
        put(place, 64, outputs[lut], slot * 4, 1.f);
    }
    extracted = aner_conv(graph, state, XCH, channels, extract);
    spread = extracted < 0 ? -1 : aner_conv(graph, extracted, 512, XCH, geom->x0);
    base = spread < 0 ? -1 : aner_const(graph, 512, low);
    step = base < 0 ? -1 : aner_const(graph, 512, delta);
    if (spread >= 0 && step >= 0)
        step = aner_mul(graph, spread, step);
    if (base >= 0 && step >= 0)
        base = aner_add(graph, base, step);
    result = base < 0 ? -1 : lerp_bits(graph, base, extracted, geom);
    placed = result < 0 ? -1 : aner_conv(graph, result, channels, 64, place);
    free(extract);
    free(low);
    free(delta);
    free(place);
    return placed;
}

static int combinational(AnerGraph *graph, int state, const Geom *geom, int channels,
                         const int32_t *inputs, const int32_t *outputs, const float *entries,
                         const int32_t *level_luts, int count) {
    float *pass = zeros((size_t)channels * (size_t)channels);
    int carried, cursor;
    if (!pass)
        return -1;
    for (int wire = 0; wire < channels; wire++)
        put(pass, channels, wire, wire, 1.f);
    for (int i = 0; i < count; i++) {
        int wire = outputs[level_luts[i]];
        for (int col = 0; col < channels; col++)
            put(pass, channels, wire, col, 0.f);
    }
    carried = aner_conv(graph, state, channels, channels, pass);
    free(pass);
    if (carried < 0)
        return -1;
    cursor = 0;
    while (cursor < count) {
        int tile = count - cursor;
        int placed;
        if (tile > TILE)
            tile = TILE;
        placed = eval_tile(graph, state, geom, channels, inputs, outputs, entries,
                           level_luts + cursor, tile);
        if (placed < 0)
            return -1;
        carried = aner_add(graph, carried, placed);
        if (carried < 0)
            return -1;
        cursor += tile;
    }
    return carried;
}

static int schedule_ok(int32_t wires, int32_t nlut, const int32_t *inputs, const int32_t *outputs,
                       const int32_t *level_luts, const int32_t *level_counts, int32_t nlevels) {
    int32_t listed = 0;
    if (nlevels < 0 || nlut < 0)
        return 0;
    if (nlut > 0 && (!inputs || !outputs))
        return 0;
    if (nlevels > 0 && (!level_luts || !level_counts))
        return 0;
    for (int32_t i = 0; i < nlut; i++) {
        if (outputs[i] < 0 || outputs[i] >= wires)
            return 0;
        for (int pin = 0; pin < 6; pin++) {
            int32_t wire = inputs[(size_t)i * 6 + pin];
            if (wire < -1 || wire >= wires)
                return 0;
        }
    }
    for (int32_t level = 0; level < nlevels; level++) {
        if (level_counts[level] < 0)
            return 0;
        listed += level_counts[level];
    }
    for (int32_t i = 0; i < listed; i++) {
        if (level_luts[i] < 0 || level_luts[i] >= nlut)
            return 0;
    }
    return 1;
}

AnerLUT6Program *aner_lut6_compile(
    int32_t wires,
    const int32_t *inputs,
    const int32_t *outputs,
    const float *entries,
    int32_t nlut,
    const int32_t *level_luts,
    const int32_t *level_counts,
    int32_t nlevels) {
    int channels = state_channels(wires);
    AnerGraph *graph;
    Geom geom;
    AnerLUT6Program *program;
    AnerProgram *compiled;
    int state;
    const int32_t *cursor;
    if (!channels || (nlut > 0 && !entries))
        return NULL;
    if (!schedule_ok(wires, nlut, inputs, outputs, level_luts, level_counts, nlevels))
        return NULL;
    graph = aner_graph_new();
    if (!graph)
        return NULL;
    if (geom_build(&geom) != 0) {
        geom_free(&geom);
        aner_graph_free(graph);
        return NULL;
    }
    state = aner_input(graph, channels);
    cursor = level_luts;
    for (int32_t level = 0; level < nlevels; level++) {
        int count = level_counts[level];
        if (count > 0) {
            state = combinational(graph, state, &geom, channels, inputs, outputs, entries, cursor, count);
            if (state < 0)
                break;
        }
        cursor += count;
    }
    geom_free(&geom);
    if (state < 0 || aner_output(graph, state) < 0) {
        aner_graph_free(graph);
        return NULL;
    }
    compiled = aner_compile(graph, 64);
    aner_graph_free(graph);
    if (!compiled)
        return NULL;
    program = calloc(1, sizeof(*program));
    if (!program) {
        aner_program_free(compiled);
        return NULL;
    }
    program->program = compiled;
    program->wires = wires;
    program->compile_ms = aner_compile_milliseconds(compiled);
    return program;
}

void aner_lut6_free(AnerLUT6Program *program) {
    if (!program)
        return;
    aner_program_free(program->program);
    free(program);
}

int aner_lut6_wires(const AnerLUT6Program *program) {
    return program ? program->wires : 0;
}

int aner_lut6_uses_neural_engine(const AnerLUT6Program *program) {
    return program && aner_uses_neural_engine(program->program);
}

double aner_lut6_compile_milliseconds(const AnerLUT6Program *program) {
    return program ? program->compile_ms : 0;
}

int aner_lut6_forward(AnerLUT6Program *program, float *wires, int32_t batch, double *exec_ms) {
    int channels, rc;
    float *packed_in, *packed_out;
    const float *inputs[1];
    float *outputs[1];
    if (!program || !wires || batch <= 0)
        return -1;
    channels = state_channels(program->wires);
    packed_in = zeros((size_t)channels * (size_t)batch);
    packed_out = zeros((size_t)channels * (size_t)batch);
    if (!packed_in || !packed_out) {
        free(packed_in);
        free(packed_out);
        return -1;
    }
    for (int32_t sample = 0; sample < batch; sample++) {
        for (int wire = 0; wire < program->wires; wire++)
            packed_in[(size_t)wire * (size_t)batch + (size_t)sample] =
                wires[(size_t)sample * (size_t)program->wires + (size_t)wire];
    }
    inputs[0] = packed_in;
    outputs[0] = packed_out;
    rc = aner_run_batch(program->program, inputs, outputs, batch, exec_ms);
    if (rc == 0) {
        for (int32_t sample = 0; sample < batch; sample++) {
            for (int wire = 0; wire < program->wires; wire++)
                wires[(size_t)sample * (size_t)program->wires + (size_t)wire] =
                    packed_out[(size_t)wire * (size_t)batch + (size_t)sample];
        }
    }
    free(packed_in);
    free(packed_out);
    return rc;
}
