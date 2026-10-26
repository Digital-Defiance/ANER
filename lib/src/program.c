#include "aner.h"
#include "internal.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char *aner_version(void) {
    return ANER_VERSION_STRING;
}

struct AnerProgram {
    AnerGraph *graph;
    int spatial;      /* ANE tile width, and the single-tile width */
    int gpu_spatial;  /* GPU chunk width; a multiple of spatial for split runs */
    AnerSchedule schedule;
    int on_ane;
    int on_gpu;
    void *ane;
    void *gpu;
    double compile_ms;
    /* Measured wall milliseconds per chunk for each engine, [0] Neural Engine tile,
       [1] GPU chunk; 0 = unknown. Seeded at compile time for split programs and
       refined during batches. The split scheduler uses them to stop an engine from
       claiming work the other engine would finish sooner. */
    _Atomic double est_ms[2];
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void update_estimate(AnerProgram *program, int engine, double sample) {
    if (sample <= 0)
        return;
    double old = atomic_load(&program->est_ms[engine]);
    atomic_store(&program->est_ms[engine], old > 0 ? 0.7 * old + 0.3 * sample : sample);
}

/* Time one Neural Engine tile and one GPU chunk on zero input (one warm-up run, one
   timed run each) so a split's very first batch already knows which engine is faster:
   a deep chain can be several times faster on the Neural Engine, a wide MLP on the GPU. */
static void calibrate_split(AnerProgram *program) {
    const AnerGraph *g = program->graph;
    int widths[2] = {program->spatial, program->gpu_spatial};
    for (int e = 0; e < 2; e++) {
        int width = widths[e];
        const float **in = calloc((size_t)(g->ninputs > 0 ? g->ninputs : 1), sizeof(float *));
        float **out = calloc((size_t)(g->noutputs > 0 ? g->noutputs : 1), sizeof(float *));
        int ok = in && out;
        for (int i = 0; ok && i < g->nops; i++) {
            const AnerOp *op = &g->ops[i];
            if (op->kind == ANER_INPUT &&
                !(in[op->input_index] = calloc((size_t)op->channels * (size_t)width, sizeof(float))))
                ok = 0;
            if (ok && op->is_output && !out[op->output_index] &&
                !(out[op->output_index] = calloc((size_t)op->channels * (size_t)width, sizeof(float))))
                ok = 0;
        }
        double timed = 0;
        for (int rep = 0; ok && rep < 2; rep++) {
            double ms = 0, t0 = now_ms();
            int rc = e == 0 ? aner_ane_eval(program->ane, in, out, &ms) : aner_gpu_eval(program->gpu, in, out, &ms);
            if (rc != 0) {
                timed = 0;
                break;
            }
            timed = now_ms() - t0;  /* the second run is the timed one */
        }
        atomic_store(&program->est_ms[e], timed);
        if (in)
            for (int i = 0; i < g->ninputs; i++)
                free((void *)in[i]);
        if (out)
            for (int i = 0; i < g->noutputs; i++)
                free(out[i]);
        free(in);
        free(out);
    }
}

/* The GPU is far more efficient on wide chunks than on one 64-column tile: a
   128-step chain runs at ~13 TFLOPS on 2048 columns but under 3 on 64, because
   each dispatch pays a fixed launch cost. So the GPU backend compiles for a
   wide spatial, the largest multiple of the caller's tile that does not exceed
   this target, and a batch is fed to it one wide chunk at a time. */
static int gpu_split_spatial(int ane_spatial) {
    /* 4096 measured best on an M4 Max for a 512-wide, 18-layer MLP (65,536 columns):
       GPU-only 1.67M columns/s, split 2.03M (2048: 1.51M / 1.86M; 16384 starves the
       Neural Engine in a split). */
    int target = 4096;
    const char *env = getenv("ANER_GPU_CHUNK");  /* tuning knob: target GPU chunk width */
    if (env && atoi(env) > 0)
        target = atoi(env);
    int width = ane_spatial;
    while (width + ane_spatial <= target)
        width += ane_spatial;
    return width;
}

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

AnerProgram *aner_compile_scheduled(const AnerGraph *graph, int spatial, AnerSchedule schedule) {
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
    /* Both GPU schedules widen the GPU's spatial: Metal is starved by 64-column
       dispatches, so one dispatch should cover many tiles' worth of columns.
       The width is a multiple of the caller's spatial, so a batch tiles evenly
       and, in a split, the GPU's chunk stays aligned with the Neural Engine's
       tiles on the shared cursor. */
    program->gpu_spatial = schedule == ANER_SCHED_ANE ? spatial : gpu_split_spatial(spatial);
    program->schedule = schedule;

    double ane_ms = 0, gpu_ms = 0;
    if (schedule == ANER_SCHED_ANE || schedule == ANER_SCHED_SPLIT) {
        program->ane = aner_ane_compile(owned, spatial, &ane_ms);
        program->on_ane = program->ane != NULL;
    }
    if (schedule == ANER_SCHED_GPU || schedule == ANER_SCHED_SPLIT) {
        program->gpu = aner_gpu_compile(owned, program->gpu_spatial, &gpu_ms);
        program->on_gpu = program->gpu != NULL;
    }
    /* Report the slower compile; both happen before the first run. */
    program->compile_ms = ane_ms > gpu_ms ? ane_ms : gpu_ms;
    atomic_store(&program->est_ms[0], 0.0);
    atomic_store(&program->est_ms[1], 0.0);
    if (schedule == ANER_SCHED_SPLIT && program->on_ane && program->on_gpu)
        calibrate_split(program);
    return program;
}

AnerProgram *aner_compile(const AnerGraph *graph, int spatial) {
    return aner_compile_scheduled(graph, spatial, ANER_SCHED_ANE);
}

void aner_program_free(AnerProgram *program) {
    if (!program)
        return;
    aner_ane_release(program->ane);
    aner_gpu_release(program->gpu);
    aner_graph_free(program->graph);
    free(program);
}

int aner_uses_neural_engine(const AnerProgram *program) {
    return program && program->on_ane;
}

int aner_uses_gpu(const AnerProgram *program) {
    return program && program->on_gpu;
}

double aner_compile_milliseconds(const AnerProgram *program) {
    return program ? program->compile_ms : 0;
}

int aner_run_reference(const AnerGraph *graph, int spatial, const float *const *inputs, float *const *outputs) {
    return aner_cpu_run(graph, spatial, inputs, outputs);
}

/* Evaluate `valid` columns on the GPU program, which is compiled for
   program->gpu_spatial. When the caller's tile is narrower (the single-tile
   path, where valid == program->spatial <= gpu_spatial), the input is
   zero-padded up to the compiled width and only the valid columns are read
   back. When valid == gpu_spatial this is a plain pass-through. */
static int gpu_eval_tile(AnerProgram *program, const float *const *inputs, float *const *outputs,
                         int valid, double *exec_ms);

int aner_run(AnerProgram *program, const float *const *inputs, float *const *outputs, double *exec_ms) {
    if (!program)
        return -1;
    if (program->on_ane) {
        int rc = aner_ane_eval(program->ane, inputs, outputs, exec_ms);
        if (rc == 0)
            return 0;
        program->on_ane = 0;
    }
    if (program->on_gpu) {
        int rc = gpu_eval_tile(program, inputs, outputs, program->spatial, exec_ms);
        if (rc == 0)
            return 0;
        program->on_gpu = 0;
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

static int gpu_eval_tile(AnerProgram *program, const float *const *inputs, float *const *outputs,
                         int valid, double *exec_ms) {
    int width = program->gpu_spatial;
    if (valid == width)
        return aner_gpu_eval(program->gpu, inputs, outputs, exec_ms);
    if (valid > width)
        return -1;
    /* Narrow tile: pad each input up to the compiled width, read back valid. */
    const AnerGraph *g = program->graph;
    int rc = 0;
    const float **pin = calloc((size_t)g->ninputs, sizeof(float *));
    float **pout = calloc((size_t)g->noutputs, sizeof(float *));
    if ((g->ninputs && !pin) || (g->noutputs && !pout)) {
        free(pin); free(pout);
        return -1;
    }
    for (int i = 0; i < g->nops && rc == 0; i++) {
        const AnerOp *op = &g->ops[i];
        if (op->kind == ANER_INPUT) {
            int idx = op->input_index;
            pin[idx] = calloc((size_t)op->channels * (size_t)width, sizeof(float));
            if (!pin[idx]) { rc = -1; break; }
            for (int c = 0; c < op->channels; c++)
                memcpy((float *)pin[idx] + (size_t)c * width,
                       inputs[idx] + (size_t)c * valid, sizeof(float) * (size_t)valid);
        }
        if (op->is_output) {
            int idx = op->output_index;
            if (!pout[idx]) {
                pout[idx] = calloc((size_t)op->channels * (size_t)width, sizeof(float));
                if (!pout[idx]) { rc = -1; break; }
            }
        }
    }
    if (rc == 0)
        rc = aner_gpu_eval(program->gpu, pin, pout, exec_ms);
    if (rc == 0) {
        for (int i = 0; i < g->nops; i++) {
            const AnerOp *op = &g->ops[i];
            if (!op->is_output)
                continue;
            int idx = op->output_index;
            for (int c = 0; c < op->channels; c++)
                memcpy(outputs[idx] + (size_t)c * valid,
                       pout[idx] + (size_t)c * width, sizeof(float) * (size_t)valid);
        }
    }
    for (int i = 0; i < g->ninputs; i++)
        free((void *)pin[i]);
    for (int i = 0; i < g->noutputs; i++)
        free(pout[i]);
    free(pin);
    free(pout);
    return rc;
}

/* Channel counts for inputs and outputs, in slot order. */
typedef struct {
    int ninputs;
    int noutputs;
    int *in_ch;
    int *out_ch;
} IOShape;

static int io_shape_init(const AnerGraph *graph, IOShape *s) {
    s->ninputs = graph->ninputs;
    s->noutputs = graph->noutputs;
    s->in_ch = calloc((size_t)graph->ninputs, sizeof(int));
    s->out_ch = calloc((size_t)graph->noutputs, sizeof(int));
    if ((graph->ninputs && !s->in_ch) || (graph->noutputs && !s->out_ch)) {
        free(s->in_ch);
        free(s->out_ch);
        return -1;
    }
    for (int i = 0; i < graph->nops; i++) {
        const AnerOp *op = &graph->ops[i];
        if (op->kind == ANER_INPUT)
            s->in_ch[op->input_index] = op->channels;
        if (op->is_output)
            s->out_ch[op->output_index] = op->channels;
    }
    return 0;
}

static void io_shape_free(IOShape *s) {
    free(s->in_ch);
    free(s->out_ch);
}

/* Per-worker scratch: its own packed tile buffers so two threads never touch
   the same memory. */
typedef struct {
    float **in;
    float **out;
} TileBuf;

static int tilebuf_init(const IOShape *s, int spatial, TileBuf *b) {
    b->in = calloc((size_t)s->ninputs, sizeof(float *));
    b->out = calloc((size_t)s->noutputs, sizeof(float *));
    if ((s->ninputs && !b->in) || (s->noutputs && !b->out))
        return -1;
    for (int i = 0; i < s->ninputs; i++) {
        b->in[i] = calloc((size_t)s->in_ch[i] * (size_t)spatial, sizeof(float));
        if (!b->in[i])
            return -1;
    }
    for (int i = 0; i < s->noutputs; i++) {
        b->out[i] = calloc((size_t)s->out_ch[i] * (size_t)spatial, sizeof(float));
        if (!b->out[i])
            return -1;
    }
    return 0;
}

static void tilebuf_free(const IOShape *s, TileBuf *b) {
    if (b->in)
        for (int i = 0; i < s->ninputs; i++)
            free(b->in[i]);
    if (b->out)
        for (int i = 0; i < s->noutputs; i++)
            free(b->out[i]);
    free(b->in);
    free(b->out);
}

typedef int (*TileEval)(void *handle, const float *const *inputs, float *const *outputs, double *exec_ms);

/* Shared across the two workers of a split run. The cursor is the next column
   not yet claimed; each worker atomically grabs a chunk of its own width. The
   ANE worker claims 64-column tiles; the GPU worker claims a wide chunk so one
   dispatch covers many tiles. Both widths are multiples of the base tile, so
   whichever worker grabs a column range, the range is tile-aligned and never
   overlaps another worker's range. */
typedef struct {
    AnerProgram *program;
    const IOShape *shape;
    const float *const *inputs;
    float *const *outputs;
    int batch;
    int nworkers;
    _Atomic int cursor;
    _Atomic int failed;
    _Atomic int active;  /* workers still claiming; the last one never declines */
} SplitJob;

typedef struct {
    SplitJob *job;
    TileEval eval;
    void *handle;
    int tile_width;   /* columns this worker claims and its engine was compiled for */
    int pipelined;    /* GPU worker: two chunks in flight via aner_gpu_submit/wait */
    int engine;       /* 0 Neural Engine, 1 GPU: index into program->est_ms */
    int other_width;  /* the other engine's chunk width */
    int forced;       /* set when this worker must finish the batch alone */
    double busy_ms;
    int tiles;
} Worker;

/* Claim this worker's next chunk only if it can finish it before the other engine could
   finish all remaining columns alone. `pending` counts this worker's chunks already in
   flight, which the new chunk would queue behind. Without estimates, claim greedily. */
static int should_claim(const Worker *w, int remaining, int pending) {
    AnerProgram *p = w->job->program;
    double mine = atomic_load(&p->est_ms[w->engine]);
    double theirs = atomic_load(&p->est_ms[1 - w->engine]);
    if (mine <= 0 || theirs <= 0)
        return 1;
    double their_time = ceil((double)remaining / (double)w->other_width) * theirs;
    return mine * (double)(1 + pending) < their_time;
}

/* Atomically claim the next chunk for w. Returns 1 with the column range, or 0 when the
   batch is done, failed, or w steps aside for the faster engine. A worker that steps
   aside leaves the batch to the others; the last active worker is never allowed to step
   aside, so every column is processed. */
static int claim_chunk(Worker *w, int pending, int *column_out, int *valid_out) {
    SplitJob *job = w->job;
    int width = w->tile_width;
    for (;;) {
        if (atomic_load(&job->failed))
            return 0;
        int c = atomic_load(&job->cursor);
        if (c >= job->batch)
            return 0;
        if (!w->forced && job->nworkers > 1 && !should_claim(w, job->batch - c, pending)) {
            if (atomic_fetch_sub(&job->active, 1) > 1)
                return 0;
            atomic_fetch_add(&job->active, 1);
            w->forced = 1;
        }
        if (atomic_compare_exchange_weak(&job->cursor, &c, c + width)) {
            int valid = job->batch - c;
            *column_out = c;
            *valid_out = valid > width ? width : valid;
            return 1;
        }
    }
}

/* One worker loop: claim a tile, pack, evaluate on this worker's engine,
   unpack. Buffers are private to the worker; inputs and outputs are shared but
   each column belongs to exactly one tile, so writes never overlap. */
static void worker_run(Worker *w) {
    SplitJob *job = w->job;
    const IOShape *s = job->shape;
    int width = w->tile_width;
    TileBuf buf;
    memset(&buf, 0, sizeof(buf));
    if (tilebuf_init(s, width, &buf) != 0) {
        atomic_store(&job->failed, 1);
        tilebuf_free(s, &buf);
        return;
    }
    for (;;) {
        int column, valid;
        if (!claim_chunk(w, 0, &column, &valid))
            break;
        double t_chunk = now_ms();
        for (int i = 0; i < s->ninputs; i++)
            pack_tile(job->inputs[i], buf.in[i], s->in_ch[i], job->batch, width, column, valid);
        double ms = 0;
        int rc = w->eval(w->handle, (const float *const *)buf.in, buf.out, &ms);
        if (rc != 0) {
            /* Engine failed this chunk. Recover on the interpreter rather than
               failing the whole batch, matching aner_run's fallback. The chunk
               is already packed to this worker's width in buf. */
            ms = 0;
            rc = aner_cpu_run(job->program->graph, width, (const float *const *)buf.in, buf.out);
            if (rc != 0) {
                atomic_store(&job->failed, 1);
                break;
            }
        }
        w->busy_ms += ms;
        w->tiles++;
        for (int i = 0; i < s->noutputs; i++)
            unpack_tile(job->outputs[i], buf.out[i], s->out_ch[i], job->batch, width, column, valid);
        if (ms > 0)
            update_estimate(job->program, w->engine, now_ms() - t_chunk);
    }
    tilebuf_free(s, &buf);
}

static int gpu_pipeline_enabled(void) {
    const char *env = getenv("ANER_GPU_PIPELINE");
    return !(env && env[0] == '0');
}

/* Wait for a submitted GPU chunk and unpack it. If the GPU reports an error,
   recompute the chunk on the interpreter from the slot's still-packed input. */
static void gpu_finish(Worker *w, float **in, float **out, int slot, int column, int valid,
                       double submitted_ms, double *last_done_ms) {
    SplitJob *job = w->job;
    const IOShape *s = job->shape;
    double ms = 0;
    int gpu_ok = aner_gpu_wait(w->handle, slot, &ms) == 0;
    if (gpu_ok) {
        /* A chunk queued behind another started when that one finished. */
        double done = now_ms();
        double start = submitted_ms > *last_done_ms ? submitted_ms : *last_done_ms;
        update_estimate(job->program, w->engine, done - start);
        *last_done_ms = done;
    }
    if (!gpu_ok) {
        ms = 0;
        if (aner_cpu_run(job->program->graph, w->tile_width, (const float *const *)in, out) != 0) {
            atomic_store(&job->failed, 1);
            return;
        }
    }
    w->busy_ms += ms;
    w->tiles++;
    for (int i = 0; i < s->noutputs; i++)
        unpack_tile(job->outputs[i], out[i], s->out_ch[i], job->batch, w->tile_width, column, valid);
}

/* GPU worker with two chunks in flight. Each iteration claims a chunk, packs it
   directly into the free slot's shared buffers (no intermediate copy), submits
   it, then waits for and unpacks the previous chunk while the new one runs. */
static void gpu_worker_pipelined(Worker *w) {
    SplitJob *job = w->job;
    const IOShape *s = job->shape;
    int width = w->tile_width;
    float **in_ptr[2] = {NULL, NULL};
    float **out_ptr[2] = {NULL, NULL};
    int pend_col[2] = {-1, -1};
    int pend_valid[2] = {0, 0};
    double pend_submit[2] = {0, 0};
    double last_done = 0;
    for (int k = 0; k < 2; k++) {
        in_ptr[k] = calloc((size_t)(s->ninputs > 0 ? s->ninputs : 1), sizeof(float *));
        out_ptr[k] = calloc((size_t)(s->noutputs > 0 ? s->noutputs : 1), sizeof(float *));
        if (!in_ptr[k] || !out_ptr[k]) {
            atomic_store(&job->failed, 1);
            goto done;
        }
        for (int i = 0; i < s->ninputs; i++)
            if (!(in_ptr[k][i] = aner_gpu_slot_input(w->handle, k, i))) {
                atomic_store(&job->failed, 1);
                goto done;
            }
        for (int i = 0; i < s->noutputs; i++)
            if (!(out_ptr[k][i] = aner_gpu_slot_output(w->handle, k, i))) {
                atomic_store(&job->failed, 1);
                goto done;
            }
    }
    int slot = 0;
    for (;;) {
        int column = -1, valid = 0;
        int pending = (pend_col[0] >= 0) + (pend_col[1] >= 0);
        if (!claim_chunk(w, pending, &column, &valid))
            column = -1;
        if (column >= 0) {
            for (int i = 0; i < s->ninputs; i++)
                pack_tile(job->inputs[i], in_ptr[slot][i], s->in_ch[i], job->batch, width, column, valid);
            if (aner_gpu_submit(w->handle, slot) == 0) {
                pend_col[slot] = column;
                pend_valid[slot] = valid;
                pend_submit[slot] = now_ms();
            } else if (aner_cpu_run(job->program->graph, width, (const float *const *)in_ptr[slot],
                                    out_ptr[slot]) == 0) {
                /* Could not launch: this chunk ran on the interpreter instead. */
                w->tiles++;
                for (int i = 0; i < s->noutputs; i++)
                    unpack_tile(job->outputs[i], out_ptr[slot][i], s->out_ch[i], job->batch, width, column, valid);
            } else {
                atomic_store(&job->failed, 1);
            }
        }
        int other = slot ^ 1;
        if (pend_col[other] >= 0) {
            gpu_finish(w, in_ptr[other], out_ptr[other], other, pend_col[other], pend_valid[other],
                       pend_submit[other], &last_done);
            pend_col[other] = -1;
        }
        if (column < 0) {
            if (pend_col[slot] >= 0) {
                gpu_finish(w, in_ptr[slot], out_ptr[slot], slot, pend_col[slot], pend_valid[slot],
                           pend_submit[slot], &last_done);
                pend_col[slot] = -1;
            }
            break;
        }
        slot = other;
    }
done:
    for (int k = 0; k < 2; k++) {
        free(in_ptr[k]);
        free(out_ptr[k]);
    }
}

static void worker_dispatch(Worker *w) {
    if (w->pipelined)
        gpu_worker_pipelined(w);
    else
        worker_run(w);
}

static void *worker_thunk(void *arg) {
    worker_dispatch((Worker *)arg);
    return NULL;
}

/* Run a wide batch across the engines this schedule enables, each taking the
   next chunk of columns as it frees up, at its own tile width. With both
   engines the faster one naturally does more work and the run finishes near
   the busier engine's time; with one engine this is a single worker tiling at
   that engine's width (a wide chunk for the GPU, a 64-column tile for the ANE).
   Returns 0, or a negative code if a chunk failed and nothing recovered it. */
static int run_concurrent_batch(AnerProgram *program, const IOShape *shape,
                                const float *const *inputs, float *const *outputs,
                                int batch, double *exec_ms) {
    SplitJob job;
    job.program = program;
    job.shape = shape;
    job.inputs = inputs;
    job.outputs = outputs;
    job.batch = batch;
    atomic_store(&job.cursor, 0);
    atomic_store(&job.failed, 0);
    job.nworkers = 0;

    Worker workers[2];
    int nworkers = 0;
    if (program->on_ane) {
        workers[nworkers].job = &job;
        workers[nworkers].eval = aner_ane_eval;
        workers[nworkers].handle = program->ane;
        workers[nworkers].tile_width = program->spatial;
        workers[nworkers].pipelined = 0;
        workers[nworkers].engine = 0;
        workers[nworkers].other_width = program->gpu_spatial;
        workers[nworkers].forced = 0;
        workers[nworkers].busy_ms = 0;
        workers[nworkers].tiles = 0;
        nworkers++;
    }
    if (program->on_gpu) {
        workers[nworkers].job = &job;
        workers[nworkers].eval = aner_gpu_eval;
        workers[nworkers].handle = program->gpu;
        workers[nworkers].tile_width = program->gpu_spatial;
        workers[nworkers].pipelined = gpu_pipeline_enabled();
        workers[nworkers].engine = 1;
        workers[nworkers].other_width = program->spatial;
        workers[nworkers].forced = 0;
        workers[nworkers].busy_ms = 0;
        workers[nworkers].tiles = 0;
        nworkers++;
    }
    if (nworkers == 0)
        return -1;
    job.nworkers = nworkers;
    atomic_store(&job.active, nworkers);

    pthread_t threads[2];
    for (int i = 1; i < nworkers; i++)
        pthread_create(&threads[i], NULL, worker_thunk, &workers[i]);
    worker_dispatch(&workers[0]);     /* run worker 0 on the calling thread */
    for (int i = 1; i < nworkers; i++)
        pthread_join(threads[i], NULL);

    if (atomic_load(&job.failed))
        return -1;
    if (exec_ms) {
        /* Wall time is dominated by the busier engine; report that. */
        double m = 0;
        for (int i = 0; i < nworkers; i++)
            if (workers[i].busy_ms > m)
                m = workers[i].busy_ms;
        *exec_ms = m;
    }
    return 0;
}

/* Serial batch: pack, run one tile through aner_run (which carries its own
   engine fallback), unpack. Used for single-engine schedules and as the
   fallback when a split cannot run concurrently. */
static int run_serial_batch(AnerProgram *program, const IOShape *shape,
                            const float *const *inputs, float *const *outputs,
                            int batch, double *exec_ms) {
    int spatial = program->spatial;
    TileBuf buf;
    memset(&buf, 0, sizeof(buf));
    if (tilebuf_init(shape, spatial, &buf) != 0) {
        tilebuf_free(shape, &buf);
        return -1;
    }
    int rc = 0;
    double total = 0;
    for (int column = 0; rc == 0 && column < batch; column += spatial) {
        int valid = batch - column;
        if (valid > spatial)
            valid = spatial;
        for (int i = 0; i < shape->ninputs; i++)
            pack_tile(inputs[i], buf.in[i], shape->in_ch[i], batch, spatial, column, valid);
        double ms = 0;
        rc = aner_run(program, (const float *const *)buf.in, buf.out, &ms);
        total += ms;
        if (rc == 0)
            for (int i = 0; i < shape->noutputs; i++)
                unpack_tile(outputs[i], buf.out[i], shape->out_ch[i], batch, spatial, column, valid);
    }
    if (exec_ms)
        *exec_ms = total;
    tilebuf_free(shape, &buf);
    return rc;
}

int aner_run_batch(AnerProgram *program, const float *const *inputs, float *const *outputs, int batch, double *exec_ms) {
    if (!program || batch <= 0)
        return -1;
    int spatial = program->spatial;
    if (batch == spatial)
        return aner_run(program, inputs, outputs, exec_ms);

    IOShape shape;
    if (io_shape_init(program->graph, &shape) != 0)
        return -1;

    /* The concurrent path tiles each engine at its own width, so it is the one
       that feeds the GPU wide chunks. Use it whenever the GPU is live (split or
       GPU-only). ANE-only stays on the serial path: its tile is the compiled
       spatial, so per-tile aner_run is already optimal and avoids thread setup.
       If the GPU fell back at compile time, on_gpu is 0 and this stays serial. */
    int rc;
    if (program->on_gpu && batch > spatial)
        rc = run_concurrent_batch(program, &shape, inputs, outputs, batch, exec_ms);
    else
        rc = run_serial_batch(program, &shape, inputs, outputs, batch, exec_ms);

    io_shape_free(&shape);
    return rc;
}
