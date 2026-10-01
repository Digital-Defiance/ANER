#import "engines.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <mach/mach_time.h>

static double ms_since(uint64_t t0) {
    static mach_timebase_info_data_t tb;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ mach_timebase_info(&tb); });
    return (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
}

static void f32_to_f16(const float *src, uint16_t *dst, int n) {
    for (int i = 0; i < n; i++) {
        __fp16 h = (__fp16)src[i];
        memcpy(dst + i, &h, sizeof(h));
    }
}

static void f16_to_f32(const uint16_t *src, float *dst, int n) {
    for (int i = 0; i < n; i++) {
        __fp16 h;
        memcpy(&h, src + i, sizeof(h));
        dst[i] = (float)h;
    }
}

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;

static int gpu_init(void) {
    static dispatch_once_t once;
    static int ok = 0;
    dispatch_once(&once, ^{
        g_device = MTLCreateSystemDefaultDevice();
        if (g_device)
            g_queue = [g_device newCommandQueue];
        ok = g_device && g_queue;
    });
    return ok ? 0 : -1;
}

static int gpu_gemm(const float *A, const float *X, float *Y, int channels, int spatial, double *ms) {
    if (gpu_init() != 0)
        return -1;
    @autoreleasepool {
        NSUInteger wBytes = (NSUInteger)channels * (NSUInteger)channels * sizeof(uint16_t);
        NSUInteger aBytes = (NSUInteger)channels * (NSUInteger)spatial * sizeof(uint16_t);
        id<MTLBuffer> wBuf = [g_device newBufferWithLength:wBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> xBuf = [g_device newBufferWithLength:aBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> yBuf = [g_device newBufferWithLength:aBytes options:MTLResourceStorageModeShared];
        if (!wBuf || !xBuf || !yBuf)
            return -2;
        f32_to_f16(A, wBuf.contents, channels * channels);
        f32_to_f16(X, xBuf.contents, channels * spatial);

        MPSMatrixDescriptor *wDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)channels
                                                                           columns:(NSUInteger)channels
                                                                          rowBytes:(NSUInteger)channels * sizeof(uint16_t)
                                                                          dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)channels
                                                                           columns:(NSUInteger)spatial
                                                                          rowBytes:(NSUInteger)spatial * sizeof(uint16_t)
                                                                          dataType:MPSDataTypeFloat16];
        MPSMatrix *left = [[MPSMatrix alloc] initWithBuffer:wBuf descriptor:wDesc];
        MPSMatrix *right = [[MPSMatrix alloc] initWithBuffer:xBuf descriptor:aDesc];
        MPSMatrix *result = [[MPSMatrix alloc] initWithBuffer:yBuf descriptor:aDesc];
        MPSMatrixMultiplication *mul = [[MPSMatrixMultiplication alloc] initWithDevice:g_device
                                                                         transposeLeft:NO
                                                                        transposeRight:NO
                                                                            resultRows:(NSUInteger)channels
                                                                         resultColumns:(NSUInteger)spatial
                                                                       interiorColumns:(NSUInteger)channels
                                                                                 alpha:1.0
                                                                                  beta:0.0];
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        [mul encodeToCommandBuffer:cb leftMatrix:left rightMatrix:right resultMatrix:result];
        uint64_t t0 = mach_absolute_time();
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error)
            return -3;
        if (ms)
            *ms = ms_since(t0);
        f16_to_f32(yBuf.contents, Y, channels * spatial);
        return 0;
    }
}

static int gpu_fused(const float *A, int channels, int depth, const float *X, float *Y, int spatial, double *ms) {
    if (gpu_init() != 0)
        return -1;
    @autoreleasepool {
        NSUInteger wCount = (NSUInteger)channels * (NSUInteger)channels;
        uint16_t *halfW = malloc(sizeof(uint16_t) * wCount);
        if (!halfW)
            return -2;
        f32_to_f16(A, halfW, (int)wCount);
        NSData *wData = [NSData dataWithBytes:halfW length:sizeof(uint16_t) * wCount];
        free(halfW);

        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        MPSGraphDevice *mpsDevice = [MPSGraphDevice deviceWithMTLDevice:g_device];
        MPSGraphTensor *x = [graph placeholderWithShape:@[@(channels), @(spatial)]
                                               dataType:MPSDataTypeFloat16
                                                   name:@"x"];
        MPSGraphTensor *y = x;
        for (int layer = 0; layer < depth; layer++) {
            MPSGraphTensor *w = [graph constantWithData:wData
                                                  shape:@[@(channels), @(channels)]
                                               dataType:MPSDataTypeFloat16];
            y = [graph matrixMultiplicationWithPrimaryTensor:w secondaryTensor:y name:nil];
        }
        MPSGraphShapedType *xType = [[MPSGraphShapedType alloc] initWithShape:@[@(channels), @(spatial)]
                                                                     dataType:MPSDataTypeFloat16];
        MPSGraphExecutable *exe = [graph compileWithDevice:mpsDevice
                                                     feeds:@{x: xType}
                                             targetTensors:@[y]
                                          targetOperations:nil
                                     compilationDescriptor:nil];
        if (!exe)
            return -3;
        exe.options = MPSGraphOptionsNone;
        [exe specializeWithDevice:mpsDevice inputTypes:@[xType] compilationDescriptor:nil];

        NSUInteger bytes = (NSUInteger)channels * (NSUInteger)spatial * sizeof(uint16_t);
        id<MTLBuffer> inBuf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        f32_to_f16(X, inBuf.contents, channels * spatial);
        MPSShape *shape = @[@(channels), @(spatial)];
        MPSGraphTensorData *inData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf shape:shape dataType:MPSDataTypeFloat16];
        MPSGraphTensorData *outData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf shape:shape dataType:MPSDataTypeFloat16];
        MPSGraphExecutableExecutionDescriptor *ed = [MPSGraphExecutableExecutionDescriptor new];
        ed.waitUntilCompleted = YES;
        uint64_t t0 = mach_absolute_time();
        [exe runWithMTLCommandQueue:g_queue inputsArray:@[inData] resultsArray:@[outData] executionDescriptor:ed];
        if (ms)
            *ms = ms_since(t0);
        f16_to_f32(outBuf.contents, Y, channels * spatial);
        return 0;
    }
}

int gpu_chain(const float *A, int channels, int depth, const float *X, float *Y, int spatial, double *ms) {
    if (depth <= 1)
        return gpu_gemm(A, X, Y, channels, spatial, ms);
    return gpu_fused(A, channels, depth, X, Y, spatial, ms);
}
