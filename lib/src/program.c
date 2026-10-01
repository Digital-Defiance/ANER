#include "aner.h"
#include "internal.h"

#include <stdlib.h>
#include <string.h>

const char *aner_version(void) {
    return ANER_VERSION_STRING;
}

struct AnerProgram {
    AnerGraph *graph;
    int spatial;
    int on_ane;
    void *ane;
    double compile_ms;
};

static int graph_ok(const AnerGraph *graph, int spatial) {
    if (!graph || spatial <= 0 || graph->ninputs < 1 || graph->noutputs < 1)
        return 0;
    for (int i = 0; i < graph->nops; i++) {
        if (graph->ops[i].channels <= 0)
            return 0;
        if ((graph->ops[i].kind == ANER_CONV || graph->ops[i].kind == ANER_CONST) && !graph->ops[i].weight)
            return 0;
    }
    return 1;
}

AnerProgram *aner_compile(const AnerGraph *graph, int spatial) {
    if (!graph_ok(graph, spatial))
        return NULL;
    AnerProgram *program = calloc(1, sizeof(AnerProgram));
    AnerGraph *owned = aner_graph_clone(graph);
    if (!program || !owned) {
        aner_graph_free(owned);
        free(program);
        return NULL;
    }
    program->graph = owned;
    program->spatial = spatial;
    program->ane = aner_ane_compile(owned, spatial, &program->compile_ms);
    program->on_ane = program->ane != NULL;
    return program;
}

void aner_program_free(AnerProgram *program) {
    if (!program)
        return;
    aner_ane_release(program->ane);
    aner_graph_free(program->graph);
    free(program);
}

int aner_uses_neural_engine(const AnerProgram *program) {
    return program && program->on_ane;
}

double aner_compile_milliseconds(const AnerProgram *program) {
    return program ? program->compile_ms : 0;
}

int aner_run_reference(const AnerGraph *graph, int spatial, const float *const *inputs, float *const *outputs) {
    return aner_cpu_run(graph, spatial, inputs, outputs);
}

int aner_run(AnerProgram *program, const float *const *inputs, float *const *outputs, double *exec_ms) {
    if (!program)
        return -1;
    if (program->on_ane) {
        int rc = aner_ane_eval(program->ane, inputs, outputs, exec_ms);
        if (rc == 0)
            return 0;
        program->on_ane = 0;
    }
    if (exec_ms)
        *exec_ms = 0;
    return aner_cpu_run(program->graph, program->spatial, inputs, outputs);
}

static void pack_tile(const float *src, float *tile, int channels, int batch, int spatial, int column0, int valid) {
    memset(tile, 0, sizeof(float) * (size_t)channels * (size_t)spatial);
    for (int c = 0; c < channels; c++)
        memcpy(tile + (size_t)c * spatial, src + (size_t)c * batch + column0, sizeof(float) * (size_t)valid);
}

static void unpack_tile(float *dst, const float *tile, int channels, int batch, int spatial, int column0, int valid) {
    for (int c = 0; c < channels; c++)
        memcpy(dst + (size_t)c * batch + column0, tile + (size_t)c * spatial, sizeof(float) * (size_t)valid);
}

int aner_run_batch(AnerProgram *program, const float *const *inputs, float *const *outputs, int batch, double *exec_ms) {
    if (!program || batch <= 0)
        return -1;
    int spatial = program->spatial;
    if (batch == spatial)
        return aner_run(program, inputs, outputs, exec_ms);

    int ninputs = program->graph->ninputs;
    int noutputs = program->graph->noutputs;
    const float **tile_in = calloc((size_t)ninputs, sizeof(float *));
    float **tile_out = calloc((size_t)noutputs, sizeof(float *));
    int *in_ch = calloc((size_t)ninputs, sizeof(int));
    int *out_ch = calloc((size_t)noutputs, sizeof(int));
    if (!tile_in || !tile_out || !in_ch || !out_ch) {
        free(tile_in); free(tile_out); free(in_ch); free(out_ch);
        return -1;
    }
    for (int i = 0; i < program->graph->nops; i++) {
        const AnerOp *op = &program->graph->ops[i];
        if (op->kind == ANER_INPUT)
            in_ch[op->input_index] = op->channels;
        if (op->is_output)
            out_ch[op->output_index] = op->channels;
    }
    int rc = 0;
    for (int i = 0; i < ninputs; i++) {
        tile_in[i] = calloc((size_t)in_ch[i] * (size_t)spatial, sizeof(float));
        if (!tile_in[i])
            rc = -1;
    }
    for (int i = 0; i < noutputs; i++) {
        tile_out[i] = calloc((size_t)out_ch[i] * (size_t)spatial, sizeof(float));
        if (!tile_out[i])
            rc = -1;
    }
    double total = 0;
    for (int column = 0; rc == 0 && column < batch; column += spatial) {
        int valid = batch - column;
        if (valid > spatial)
            valid = spatial;
        for (int i = 0; i < ninputs; i++)
            pack_tile(inputs[i], (float *)tile_in[i], in_ch[i], batch, spatial, column, valid);
        double ms = 0;
        rc = aner_run(program, tile_in, tile_out, &ms);
        total += ms;
        if (rc == 0) {
            for (int i = 0; i < noutputs; i++)
                unpack_tile(outputs[i], tile_out[i], out_ch[i], batch, spatial, column, valid);
        }
    }
    if (exec_ms)
        *exec_ms = total;
    for (int i = 0; i < ninputs; i++)
        free((void *)tile_in[i]);
    for (int i = 0; i < noutputs; i++)
        free(tile_out[i]);
    free(tile_in);
    free(tile_out);
    free(in_ch);
    free(out_ch);
    return rc;
}
