#include "internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <mach/mach_time.h>
#import <stdio.h>
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
   0 keeps this half on the GPU so the two engines overlap. */

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
@property (nonatomic, strong) NSArray<id<MTLBuffer>> *inBufs;
@property (nonatomic, strong) NSArray<id<MTLBuffer>> *outBufs;
@property (nonatomic, strong) NSArray<MPSGraphTensorData *> *inData;
@property (nonatomic, strong) NSArray<MPSGraphTensorData *> *outData;
@property (nonatomic, strong) MPSGraphExecutableExecutionDescriptor *ed;
@end

@implementation ANERGpuProgram
- (void)dealloc {
    free(inChannels);
    free(outChannels);
}
@end

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
                break;
            }
            case ANER_CONST: {
                /* One value per channel, broadcast across every column. */
                NSData *wData = fp16_weights(op->weight, op->channels);
                MPSGraphTensor *col = [g constantWithData:wData shape:@[@(op->channels), @1]
                                                 dataType:MPSDataTypeFloat16];
                out = [g broadcastTensor:col toShape:shape name:nil];
                break;
            }
            case ANER_CONV: {
                NSData *wData = fp16_weights(op->weight, op->channels * op->in_channels);
                MPSGraphTensor *w = [g constantWithData:wData shape:@[@(op->channels), @(op->in_channels)]
                                               dataType:MPSDataTypeFloat16];
                out = [g matrixMultiplicationWithPrimaryTensor:w secondaryTensor:val[op->a] name:nil];
                break;
            }
            case ANER_ADD:
                out = [g additionWithPrimaryTensor:val[op->a] secondaryTensor:val[op->b] name:nil];
                break;
            case ANER_MUL:
                out = [g multiplicationWithPrimaryTensor:val[op->a] secondaryTensor:val[op->b] name:nil];
                break;
            case ANER_SIGMOID:
                out = [g sigmoidWithTensor:val[op->a] name:nil];
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
                    [targets addObject:[g castTensor:val[i] toType:MPSDataTypeFloat32 name:nil]];
                    break;
                }
            }
        }

        MPSGraphExecutable *exe = nil;
        if (ok) {
            MPSGraphCompilationDescriptor *cd = [MPSGraphCompilationDescriptor new];
            cd.optimizationLevel = MPSGraphOptimizationLevel0;
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
        NSMutableArray<id<MTLBuffer>> *inBufs = [NSMutableArray array];
        NSMutableArray<id<MTLBuffer>> *outBufs = [NSMutableArray array];
        NSMutableArray<MPSGraphTensorData *> *inData = [NSMutableArray array];
        NSMutableArray<MPSGraphTensorData *> *outData = [NSMutableArray array];
        for (int i = 0; i < program->ninputs; i++) {
            NSUInteger bytes = (NSUInteger)program->inChannels[i] * (NSUInteger)spatial * sizeof(float);
            id<MTLBuffer> buf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            [inBufs addObject:buf];
            [inData addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                     shape:@[@(program->inChannels[i]), @(spatial)]
                                                                  dataType:MPSDataTypeFloat32]];
        }
        for (int i = 0; i < program->noutputs; i++) {
            NSUInteger bytes = (NSUInteger)program->outChannels[i] * (NSUInteger)spatial * sizeof(float);
            id<MTLBuffer> buf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            [outBufs addObject:buf];
            [outData addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                      shape:@[@(program->outChannels[i]), @(spatial)]
                                                                   dataType:MPSDataTypeFloat32]];
        }
        program.inBufs = inBufs;
        program.outBufs = outBufs;
        program.inData = inData;
        program.outData = outData;
        program.ed = [MPSGraphExecutableExecutionDescriptor new];
        program.ed.waitUntilCompleted = YES;
        return (void *)CFBridgingRetain(program);
    }
}

int aner_gpu_eval(void *handle, const float *const *inputs, float *const *outputs, double *exec_ms) {
    ANERGpuProgram *program = (__bridge ANERGpuProgram *)handle;
    if (!program)
        return -1;
    @autoreleasepool {
        for (int i = 0; i < program->ninputs; i++) {
            NSUInteger bytes = (NSUInteger)program->inChannels[i] * (NSUInteger)program->spatial * sizeof(float);
            memcpy(program.inBufs[i].contents, inputs[i], bytes);
        }
        uint64_t t0 = mach_absolute_time();
        @try {
            [program.exe runWithMTLCommandQueue:g_queue inputsArray:program.inData
                                   resultsArray:program.outData executionDescriptor:program.ed];
        } @catch (NSException *ex) {
            fprintf(stderr, "GPU graph eval failed: %s\n", ex.reason.UTF8String);
            return -1;
        }
        if (exec_ms)
            *exec_ms = ms_since(t0);
        for (int i = 0; i < program->noutputs; i++) {
            NSUInteger bytes = (NSUInteger)program->outChannels[i] * (NSUInteger)program->spatial * sizeof(float);
            memcpy(outputs[i], program.outBufs[i].contents, bytes);
        }
        return 0;
    }
}

void aner_gpu_release(void *handle) {
    if (handle)
        CFRelease(handle);
}
