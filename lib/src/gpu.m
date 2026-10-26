#include "internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <mach/mach_time.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>

/* Metal backend for ANER graphs.

   The whole graph is lowered to one MPSGraph executable. Each value is a
   [channels, spatial] fp16 tensor, matching the ANE path's on-chip format, so
   a split batch that sends some tiles to the Neural Engine and others here
   produces the same values to fp16 rounding. The boundary stays fp32: inputs
   are cast up on entry, outputs cast back on exit.

   The executable is compiled at MPSGraphOptimizationLevel0. Level 1 (the
   default) runs a placement pass that dispatches fp16 matmul chains onto the
   Neural Engine, which is exactly the engine we are trying to run beside. Level
   0 keeps this half on the GPU so the two engines overlap. ANER_GPU_OPT_LEVEL=1
   selects level 1 anyway, as an experiment.

   Lowering keeps the op count low, since every MPSGraph op at level 0 is its
   own kernel and memory pass:
   - A const is a [channels, 1] column, never materialized at full width. Every
     ANER op treats columns independently, so a value computed only from consts
     stays a column; add and mul broadcast it against full tensors for free, and
     an output that is still a column is broadcast once at the end.
   - A conv whose square matrix is diagonal (aner_scale, and the scale inside
     aner_gelu) becomes an elementwise multiply by a [channels, 1] column instead
     of a dense matmul.

   Each compiled program owns ANER_GPU_SLOTS sets of host-visible buffers. The
   synchronous aner_gpu_eval uses slot 0. The pipelined calls (submit / wait)
   let the batch scheduler pack the next chunk straight into a free slot while
   the GPU computes the current one. */

#define ANER_GPU_SLOTS 2

static double ms_since(uint64_t t0) {
    static mach_timebase_info_data_t tb;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ mach_timebase_info(&tb); });
    return (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
}

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static MPSGraphDevice *g_graphDevice;

static int gpu_init(void) {
    static dispatch_once_t once;
    static int ok;
    dispatch_once(&once, ^{
        g_device = MTLCreateSystemDefaultDevice();
        if (g_device) {
            g_queue = [g_device newCommandQueue];
            g_graphDevice = [MPSGraphDevice deviceWithMTLDevice:g_device];
        }
        ok = g_device && g_queue && g_graphDevice;
    });
    return ok ? 0 : -1;
}

static NSData *fp16_weights(const float *w, int count) {
    uint16_t *half = malloc(sizeof(uint16_t) * (size_t)count);
    if (!half)
        return nil;
    for (int k = 0; k < count; k++) {
        __fp16 h = (__fp16)w[k];
        memcpy(half + k, &h, sizeof(h));
    }
    return [NSData dataWithBytesNoCopy:half length:sizeof(uint16_t) * (size_t)count freeWhenDone:YES];
}

static MPSGraphOptimization gpu_opt_level(void) {
    const char *s = getenv("ANER_GPU_OPT_LEVEL");
    return (s && atoi(s) >= 1) ? MPSGraphOptimizationLevel1 : MPSGraphOptimizationLevel0;
}

static int is_diagonal(const float *w, int n) {
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            if (r != c && w[(size_t)r * (size_t)n + (size_t)c] != 0.0f)
                return 0;
    return 1;
}

/* One set of host-visible buffers plus the state of an asynchronous run on it. */
@interface ANERGpuSlot : NSObject {
@public
    uint64_t t0;
    double ms;
    volatile int error;
    int inflight;
}
@property (nonatomic, strong) NSArray<id<MTLBuffer>> *inBufs;
@property (nonatomic, strong) NSArray<id<MTLBuffer>> *outBufs;
@property (nonatomic, strong) NSArray<MPSGraphTensorData *> *inData;
@property (nonatomic, strong) NSArray<MPSGraphTensorData *> *outData;
@property (nonatomic, strong) MPSGraphExecutableExecutionDescriptor *ed;
@property (nonatomic, strong) dispatch_semaphore_t done;
@end

@implementation ANERGpuSlot
@end

/* ObjC objects live in these collections so ARC manages them; the C struct
   keeps only scalars and plain C arrays. */
@interface ANERGpuProgram : NSObject {
@public
    int spatial;
    int ninputs;
    int noutputs;
    int *inChannels;
    int *outChannels;
}
@property (nonatomic, strong) MPSGraphExecutable *exe;
@property (nonatomic, strong) NSArray<ANERGpuSlot *> *slots;
@property (nonatomic, strong) MPSGraphExecutableExecutionDescriptor *syncEd;
@end

@implementation ANERGpuProgram
- (void)dealloc {
    free(inChannels);
    free(outChannels);
}
@end

static ANERGpuSlot *make_slot(ANERGpuProgram *program) {
    ANERGpuSlot *slot = [ANERGpuSlot new];
    int spatial = program->spatial;
    NSMutableArray<id<MTLBuffer>> *inBufs = [NSMutableArray array];
    NSMutableArray<id<MTLBuffer>> *outBufs = [NSMutableArray array];
    NSMutableArray<MPSGraphTensorData *> *inData = [NSMutableArray array];
    NSMutableArray<MPSGraphTensorData *> *outData = [NSMutableArray array];
    for (int i = 0; i < program->ninputs; i++) {
        NSUInteger bytes = (NSUInteger)program->inChannels[i] * (NSUInteger)spatial * sizeof(float);
        id<MTLBuffer> buf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf)
            return nil;
        [inBufs addObject:buf];
        [inData addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                 shape:@[@(program->inChannels[i]), @(spatial)]
                                                              dataType:MPSDataTypeFloat32]];
    }
    for (int i = 0; i < program->noutputs; i++) {
        NSUInteger bytes = (NSUInteger)program->outChannels[i] * (NSUInteger)spatial * sizeof(float);
        id<MTLBuffer> buf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf)
            return nil;
        [outBufs addObject:buf];
        [outData addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                  shape:@[@(program->outChannels[i]), @(spatial)]
                                                               dataType:MPSDataTypeFloat32]];
    }
    slot.inBufs = inBufs;
    slot.outBufs = outBufs;
    slot.inData = inData;
    slot.outData = outData;
    slot.done = dispatch_semaphore_create(0);
    MPSGraphExecutableExecutionDescriptor *ed = [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = NO;
    __weak ANERGpuSlot *weakSlot = slot;
    ed.completionHandler = ^(NSArray<MPSGraphTensorData *> *results, NSError *error) {
        (void)results;
        ANERGpuSlot *strong = weakSlot;
        if (!strong)
            return;
        strong->ms = ms_since(strong->t0);
        strong->error = error != nil;
        dispatch_semaphore_signal(strong.done);
    };
    slot.ed = ed;
    return slot;
}

void *aner_gpu_compile(const AnerGraph *graph, int spatial, double *compile_ms) {
    if (compile_ms)
        *compile_ms = 0;
    if (!graph || spatial <= 0 || gpu_init() != 0)
        return NULL;
    @autoreleasepool {
        MPSGraph *g = [MPSGraph new];
        g.options = MPSGraphOptionsNone;
        /* Value tensor per op, in op order. Held in an array so ARC owns them. */
        NSMutableArray<MPSGraphTensor *> *val = [NSMutableArray arrayWithCapacity:(NSUInteger)graph->nops];
        for (int i = 0; i < graph->nops; i++)
            [val addObject:(id)[NSNull null]];
        /* colconst[i]: value i is a [channels, 1] column (computed from consts only). */
        int *colconst = calloc((size_t)graph->nops, sizeof(int));
        if (!colconst)
            return NULL;
        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds = [NSMutableDictionary dictionary];
        NSMutableArray<MPSGraphShapedType *> *feedOrder = [NSMutableArray arrayWithCapacity:(NSUInteger)graph->ninputs];
        for (int i = 0; i < graph->ninputs; i++)
            [feedOrder addObject:(id)[NSNull null]];
        int ok = 1;

        for (int i = 0; i < graph->nops && ok; i++) {
            const AnerOp *op = &graph->ops[i];
            NSArray<NSNumber *> *shape = @[@(op->channels), @(spatial)];
            MPSGraphTensor *out = nil;
            switch (op->kind) {
            case ANER_INPUT: {
                MPSGraphTensor *x = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32
                                                       name:[NSString stringWithFormat:@"in%d", op->input_index]];
                MPSGraphShapedType *t = [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeFloat32];
                feeds[x] = t;
                feedOrder[op->input_index] = t;
                out = [g castTensor:x toType:MPSDataTypeFloat16 name:nil];
                colconst[i] = 0;
                break;
            }
            case ANER_CONST: {
                /* One value per channel, kept as a column; consumers broadcast it. */
                NSData *wData = fp16_weights(op->weight, op->channels);
                if (!wData) { ok = 0; break; }
                out = [g constantWithData:wData shape:@[@(op->channels), @1] dataType:MPSDataTypeFloat16];
                colconst[i] = 1;
                break;
            }
            case ANER_CONV: {
                int n = op->channels;
                if (n == op->in_channels && is_diagonal(op->weight, n)) {
                    /* Diagonal map: elementwise scale by a column, not a dense matmul. */
                    float *d = malloc(sizeof(float) * (size_t)n);
                    if (!d) { ok = 0; break; }
                    for (int k = 0; k < n; k++)
                        d[k] = op->weight[(size_t)k * (size_t)n + (size_t)k];
                    NSData *dData = fp16_weights(d, n);
                    free(d);
                    if (!dData) { ok = 0; break; }
                    MPSGraphTensor *col = [g constantWithData:dData shape:@[@(n), @1] dataType:MPSDataTypeFloat16];
                    out = [g multiplicationWithPrimaryTensor:val[op->a] secondaryTensor:col name:nil];
                } else {
                    NSData *wData = fp16_weights(op->weight, op->channels * op->in_channels);
                    if (!wData) { ok = 0; break; }
                    MPSGraphTensor *w = [g constantWithData:wData shape:@[@(op->channels), @(op->in_channels)]
                                                   dataType:MPSDataTypeFloat16];
                    out = [g matrixMultiplicationWithPrimaryTensor:w secondaryTensor:val[op->a] name:nil];
                }
                colconst[i] = colconst[op->a];
                break;
            }
            case ANER_ADD:
                out = [g additionWithPrimaryTensor:val[op->a] secondaryTensor:val[op->b] name:nil];
                colconst[i] = colconst[op->a] && colconst[op->b];
                break;
            case ANER_MUL:
                out = [g multiplicationWithPrimaryTensor:val[op->a] secondaryTensor:val[op->b] name:nil];
                colconst[i] = colconst[op->a] && colconst[op->b];
                break;
            case ANER_SIGMOID:
                out = [g sigmoidWithTensor:val[op->a] name:nil];
                colconst[i] = colconst[op->a];
                break;
            default:
                ok = 0;
                break;
            }
            if (!out)
                ok = 0;
            else
                val[i] = out;
        }

        NSMutableArray<MPSGraphTensor *> *targets = [NSMutableArray array];
        for (int slot = 0; slot < graph->noutputs && ok; slot++) {
            for (int i = 0; i < graph->nops; i++) {
                if (graph->ops[i].is_output && graph->ops[i].output_index == slot) {
                    MPSGraphTensor *t = val[i];
                    if (colconst[i])
                        t = [g broadcastTensor:t toShape:@[@(graph->ops[i].channels), @(spatial)] name:nil];
                    [targets addObject:[g castTensor:t toType:MPSDataTypeFloat32 name:nil]];
                    break;
                }
            }
        }
        free(colconst);

        MPSGraphExecutable *exe = nil;
        if (ok) {
            MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
            cd.optimizationLevel = gpu_opt_level();
            uint64_t t0 = mach_absolute_time();
            @try {
                exe = [g compileWithDevice:g_graphDevice feeds:feeds targetTensors:targets
                          targetOperations:nil compilationDescriptor:cd];
                exe.options = MPSGraphOptionsNone;
                [exe specializeWithDevice:g_graphDevice inputTypes:feedOrder compilationDescriptor:cd];
            } @catch (NSException *ex) {
                fprintf(stderr, "GPU graph compile failed: %s\n", ex.reason.UTF8String);
                exe = nil;
            }
            if (compile_ms)
                *compile_ms = ms_since(t0);
        }
        if (!exe)
            return NULL;

        ANERGpuProgram *program = [ANERGpuProgram new];
        program.exe = exe;
        program->spatial = spatial;
        program->ninputs = graph->ninputs;
        program->noutputs = graph->noutputs;
        program->inChannels = calloc((size_t)graph->ninputs, sizeof(int));
        program->outChannels = calloc((size_t)graph->noutputs, sizeof(int));
        for (int i = 0; i < graph->nops; i++) {
            if (graph->ops[i].kind == ANER_INPUT)
                program->inChannels[graph->ops[i].input_index] = graph->ops[i].channels;
            if (graph->ops[i].is_output)
                program->outChannels[graph->ops[i].output_index] = graph->ops[i].channels;
        }
        NSMutableArray<ANERGpuSlot *> *slots = [NSMutableArray array];
        for (int s = 0; s < ANER_GPU_SLOTS; s++) {
            ANERGpuSlot *slot = make_slot(program);
            if (!slot)
                return NULL;
            [slots addObject:slot];
        }
        program.slots = slots;
        program.syncEd = [MPSGraphExecutableExecutionDescriptor new];
        program.syncEd.waitUntilCompleted = YES;
        return (void *)CFBridgingRetain(program);
    }
}

int aner_gpu_eval(void *handle, const float *const *inputs, float *const *outputs, double *exec_ms) {
    ANERGpuProgram *program = (__bridge ANERGpuProgram *)handle;
    if (!program)
        return -1;
    @autoreleasepool {
        ANERGpuSlot *slot = program.slots[0];
        for (int i = 0; i < program->ninputs; i++) {
            NSUInteger bytes = (NSUInteger)program->inChannels[i] * (NSUInteger)program->spatial * sizeof(float);
            memcpy(slot.inBufs[i].contents, inputs[i], bytes);
        }
        uint64_t t0 = mach_absolute_time();
        @try {
            [program.exe runWithMTLCommandQueue:g_queue inputsArray:slot.inData
                                   resultsArray:slot.outData executionDescriptor:program.syncEd];
        } @catch (NSException *ex) {
            fprintf(stderr, "GPU graph eval failed: %s\n", ex.reason.UTF8String);
            return -1;
        }
        if (exec_ms)
            *exec_ms = ms_since(t0);
        for (int i = 0; i < program->noutputs; i++) {
            NSUInteger bytes = (NSUInteger)program->outChannels[i] * (NSUInteger)program->spatial * sizeof(float);
            memcpy(outputs[i], slot.outBufs[i].contents, bytes);
        }
        return 0;
    }
}

float *aner_gpu_slot_input(void *handle, int slot, int input_index) {
    ANERGpuProgram *program = (__bridge ANERGpuProgram *)handle;
    if (!program || slot < 0 || slot >= ANER_GPU_SLOTS || input_index < 0 || input_index >= program->ninputs)
        return NULL;
    return (float *)program.slots[slot].inBufs[input_index].contents;
}

float *aner_gpu_slot_output(void *handle, int slot, int output_index) {
    ANERGpuProgram *program = (__bridge ANERGpuProgram *)handle;
    if (!program || slot < 0 || slot >= ANER_GPU_SLOTS || output_index < 0 || output_index >= program->noutputs)
        return NULL;
    return (float *)program.slots[slot].outBufs[output_index].contents;
}

int aner_gpu_submit(void *handle, int slot) {
    ANERGpuProgram *program = (__bridge ANERGpuProgram *)handle;
    if (!program || slot < 0 || slot >= ANER_GPU_SLOTS)
        return -1;
    ANERGpuSlot *s = program.slots[slot];
    if (s->inflight)
        return -1;
    @autoreleasepool {
        s->error = 0;
        s->ms = 0;
        s->t0 = mach_absolute_time();
        s->inflight = 1;
        @try {
            [program.exe runAsyncWithMTLCommandQueue:g_queue inputsArray:s.inData
                                        resultsArray:s.outData executionDescriptor:s.ed];
        } @catch (NSException *ex) {
            fprintf(stderr, "GPU graph async eval failed: %s\n", ex.reason.UTF8String);
            s->inflight = 0;
            return -1;
        }
    }
    return 0;
}

int aner_gpu_wait(void *handle, int slot, double *exec_ms) {
    ANERGpuProgram *program = (__bridge ANERGpuProgram *)handle;
    if (!program || slot < 0 || slot >= ANER_GPU_SLOTS)
        return -1;
    ANERGpuSlot *s = program.slots[slot];
    if (!s->inflight)
        return -1;
    dispatch_semaphore_wait(s.done, DISPATCH_TIME_FOREVER);
    s->inflight = 0;
    if (exec_ms)
        *exec_ms = s->ms;
    return s->error ? -1 : 0;
}

void aner_gpu_release(void *handle) {
    if (handle)
        CFRelease(handle);
}
