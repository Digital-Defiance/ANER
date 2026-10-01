#include "aner.h"
#include "internal.h"

#include <stdlib.h>
#include <string.h>

static int add_op(AnerGraph *graph, AnerOp op) {
    if (graph->nops == graph->cap) {
        int cap = graph->cap ? graph->cap * 2 : 16;
        AnerOp *grown = realloc(graph->ops, sizeof(AnerOp) * (size_t)cap);
        if (!grown)
            return -1;
        graph->ops = grown;
        graph->cap = cap;
    }
    graph->ops[graph->nops] = op;
    return graph->nops++;
}

static int valid_tensor(const AnerGraph *graph, int id) {
    return graph && id >= 0 && id < graph->nops;
}

AnerGraph *aner_graph_new(void) {
    AnerGraph *graph = calloc(1, sizeof(AnerGraph));
    return graph;
}

void aner_graph_free(AnerGraph *graph) {
    if (!graph)
        return;
    for (int i = 0; i < graph->nops; i++)
        free(graph->ops[i].weight);
    free(graph->ops);
    free(graph);
}

AnerGraph *aner_graph_clone(const AnerGraph *graph) {
    if (!graph)
        return NULL;
    AnerGraph *copy = aner_graph_new();
    if (!copy)
        return NULL;
    for (int i = 0; i < graph->nops; i++) {
        AnerOp op = graph->ops[i];
        op.weight = NULL;
        if (graph->ops[i].weight) {
            size_t n = (size_t)op.channels * (size_t)op.in_channels;
            op.weight = malloc(sizeof(float) * n);
            if (!op.weight) {
                aner_graph_free(copy);
                return NULL;
            }
            memcpy(op.weight, graph->ops[i].weight, sizeof(float) * n);
        }
        if (add_op(copy, op) < 0) {
            free(op.weight);
            aner_graph_free(copy);
            return NULL;
        }
    }
    copy->ninputs = graph->ninputs;
    copy->noutputs = graph->noutputs;
    return copy;
}

int aner_input(AnerGraph *graph, int channels) {
    if (!graph || channels <= 0)
        return -1;
    AnerOp op;
    memset(&op, 0, sizeof(op));
    op.kind = ANER_INPUT;
    op.channels = channels;
    op.a = -1;
    op.b = -1;
    op.input_index = graph->ninputs;
    int id = add_op(graph, op);
    if (id >= 0)
        graph->ninputs++;
    return id;
}

int aner_const(AnerGraph *graph, int channels, const float *per_channel) {
    if (!graph || channels <= 0 || !per_channel)
        return -1;
    AnerOp op;
    memset(&op, 0, sizeof(op));
    op.kind = ANER_CONST;
    op.a = -1;
    op.b = -1;
    op.channels = channels;
    op.in_channels = 1;
    op.weight = malloc(sizeof(float) * (size_t)channels);
    if (!op.weight)
        return -1;
    memcpy(op.weight, per_channel, sizeof(float) * (size_t)channels);
    int id = add_op(graph, op);
    if (id < 0)
        free(op.weight);
    return id;
}

int aner_conv(AnerGraph *graph, int input, int out_channels, int in_channels, const float *row_major) {
    if (!valid_tensor(graph, input) || out_channels <= 0 || in_channels <= 0 || !row_major)
        return -1;
    if (graph->ops[input].channels != in_channels)
        return -1;
    AnerOp op;
    memset(&op, 0, sizeof(op));
    op.kind = ANER_CONV;
    op.a = input;
    op.b = -1;
    op.channels = out_channels;
    op.in_channels = in_channels;
    size_t n = (size_t)out_channels * (size_t)in_channels;
    op.weight = malloc(sizeof(float) * n);
    if (!op.weight)
        return -1;
    memcpy(op.weight, row_major, sizeof(float) * n);
    int id = add_op(graph, op);
    if (id < 0)
        free(op.weight);
    return id;
}

static int binary(AnerGraph *graph, AnerOpKind kind, int a, int b) {
    if (!valid_tensor(graph, a) || !valid_tensor(graph, b))
        return -1;
    if (graph->ops[a].channels != graph->ops[b].channels)
        return -1;
    AnerOp op;
    memset(&op, 0, sizeof(op));
    op.kind = kind;
    op.a = a;
    op.b = b;
    op.channels = graph->ops[a].channels;
    return add_op(graph, op);
}

int aner_scale(AnerGraph *graph, int input, const float *per_channel) {
    if (!valid_tensor(graph, input) || !per_channel)
        return -1;
    int n = graph->ops[input].channels;
    float *W = calloc((size_t)n * (size_t)n, sizeof(float));
    if (!W)
        return -1;
    for (int i = 0; i < n; i++)
        W[i * n + i] = per_channel[i];
    int id = aner_conv(graph, input, n, n, W);
    free(W);
    return id;
}

int aner_repeat(AnerGraph *graph, int input, int steps, const float *square_row_major) {
    if (!valid_tensor(graph, input) || steps < 1 || !square_row_major)
        return -1;
    int n = graph->ops[input].channels;
    int cur = input;
    for (int step = 0; step < steps; step++) {
        cur = aner_conv(graph, cur, n, n, square_row_major);
        if (cur < 0)
            return -1;
    }
    return cur;
}

int aner_add(AnerGraph *graph, int a, int b) { return binary(graph, ANER_ADD, a, b); }
int aner_mul(AnerGraph *graph, int a, int b) { return binary(graph, ANER_MUL, a, b); }

int aner_sigmoid(AnerGraph *graph, int input) {
    if (!valid_tensor(graph, input))
        return -1;
    AnerOp op;
    memset(&op, 0, sizeof(op));
    op.kind = ANER_SIGMOID;
    op.a = input;
    op.b = -1;
    op.channels = graph->ops[input].channels;
    return add_op(graph, op);
}

int aner_horner(AnerGraph *graph, int input, const float *coeffs_low_to_high, int ncoeff) {
    if (!valid_tensor(graph, input) || !coeffs_low_to_high || ncoeff < 1)
        return -1;
    int n = graph->ops[input].channels;
    float *fill = malloc(sizeof(float) * (size_t)n);
    if (!fill)
        return -1;
    for (int c = 0; c < n; c++)
        fill[c] = coeffs_low_to_high[ncoeff - 1];
    int acc = aner_const(graph, n, fill);
    for (int k = ncoeff - 2; k >= 0 && acc >= 0; k--) {
        acc = aner_mul(graph, acc, input);
        if (acc < 0)
            break;
        for (int c = 0; c < n; c++)
            fill[c] = coeffs_low_to_high[k];
        int coeff = aner_const(graph, n, fill);
        if (coeff < 0) {
            acc = -1;
            break;
        }
        acc = aner_add(graph, acc, coeff);
    }
    free(fill);
    return acc;
}

int aner_gelu(AnerGraph *graph, int input) {
    if (!valid_tensor(graph, input))
        return -1;
    int n = graph->ops[input].channels;
    float *scale = malloc(sizeof(float) * (size_t)n);
    if (!scale)
        return -1;
    for (int i = 0; i < n; i++)
        scale[i] = 1.702f;
    int scaled = aner_scale(graph, input, scale);
    free(scale);
    if (scaled < 0)
        return -1;
    int sig = aner_sigmoid(graph, scaled);
    if (sig < 0)
        return -1;
    return aner_mul(graph, input, sig);
}

int aner_output(AnerGraph *graph, int tensor) {
    if (!valid_tensor(graph, tensor))
        return -1;
    if (graph->ops[tensor].is_output)
        return graph->ops[tensor].output_index;
    graph->ops[tensor].is_output = 1;
    graph->ops[tensor].output_index = graph->noutputs;
    return graph->noutputs++;
}
