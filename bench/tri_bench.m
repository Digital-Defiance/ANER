#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Accelerate/Accelerate.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
}

static const int kWarm = 10;
static const int kIters = 30;

typedef struct {
    int c, s, d;
} Config;

static int cmpDouble(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmpDouble);
    if (n & 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double gflops(int c, int s, int d) {
    return 2.0 * (double)c * (double)c * (double)s * (double)d / 1e9;
}

static NSData *fp16Weights(int ch) {
    NSUInteger n = (NSUInteger)ch * (NSUInteger)ch;
    uint16_t *p = calloc(n, sizeof(uint16_t));
    for (NSUInteger i = 0; i < n; i++)
        p[i] = (uint16_t)((arc4random() & 0x03FF) | 0x2000);
    return [NSData dataWithBytesNoCopy:p length:n * sizeof(uint16_t) freeWhenDone:YES];
}

static double benchGraph(id<MTLDevice> device, MPSGraphDevice *mpsDevice, id<MTLCommandQueue> queue,
                         int ch, int sp, int depth) {
    @autoreleasepool {
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        MPSGraphTensor *x = [graph placeholderWithShape:@[@(ch), @(sp)]
                                               dataType:MPSDataTypeFloat16
                                                   name:@"x"];
        MPSGraphTensor *y = x;
        for (int i = 0; i < depth; i++) {
            MPSGraphTensor *wt = [graph constantWithData:fp16Weights(ch)
                                                   shape:@[@(ch), @(ch)]
                                                dataType:MPSDataTypeFloat16];
            y = [graph matrixMultiplicationWithPrimaryTensor:wt secondaryTensor:y name:nil];
        }
        MPSGraphShapedType *xType = [[MPSGraphShapedType alloc] initWithShape:@[@(ch), @(sp)]
                                                                     dataType:MPSDataTypeFloat16];
        MPSGraphExecutable *exe = [graph compileWithDevice:mpsDevice
                                                     feeds:@{x: xType}
                                             targetTensors:@[y]
                                          targetOperations:nil
                                     compilationDescriptor:nil];
        if (!exe) return -1;
        exe.options = MPSGraphOptionsNone;
        [exe specializeWithDevice:mpsDevice inputTypes:@[xType] compilationDescriptor:nil];

        NSUInteger bytes = (NSUInteger)ch * (NSUInteger)sp * sizeof(uint16_t);
        id<MTLBuffer> inBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        memset(inBuf.contents, 0x3C, bytes);
        MPSShape *shape = @[@(ch), @(sp)];
        MPSGraphTensorData *inData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf shape:shape dataType:MPSDataTypeFloat16];
        MPSGraphTensorData *outData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf shape:shape dataType:MPSDataTypeFloat16];
        MPSGraphExecutableExecutionDescriptor *ed = [MPSGraphExecutableExecutionDescriptor new];
        ed.waitUntilCompleted = YES;

        for (int i = 0; i < kWarm; i++)
            [exe runWithMTLCommandQueue:queue inputsArray:@[inData] resultsArray:@[outData] executionDescriptor:ed];

        double samples[64];
        for (int i = 0; i < kIters; i++) {
            uint64_t t0 = mach_absolute_time();
            [exe runWithMTLCommandQueue:queue inputsArray:@[inData] resultsArray:@[outData] executionDescriptor:ed];
            samples[i] = ticksToMs(mach_absolute_time() - t0);
        }
        return median(samples, kIters);
    }
}

static void benchMPS(id<MTLDevice> device, id<MTLCommandQueue> queue, int ch, int sp, int depth,
                     double *wallMs, double *gpuMs) {
    @autoreleasepool {
        MPSMatrixMultiplication *mul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                         transposeLeft:NO
                                                                        transposeRight:NO
                                                                            resultRows:(NSUInteger)ch
                                                                         resultColumns:(NSUInteger)sp
                                                                       interiorColumns:(NSUInteger)ch
                                                                                 alpha:1.0
                                                                                  beta:0.0];
        MPSMatrixDescriptor *wDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)ch
                                                                           columns:(NSUInteger)ch
                                                                          rowBytes:(NSUInteger)ch * sizeof(uint16_t)
                                                                          dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)ch
                                                                           columns:(NSUInteger)sp
                                                                          rowBytes:(NSUInteger)sp * sizeof(uint16_t)
                                                                          dataType:MPSDataTypeFloat16];
        NSUInteger wBytes = (NSUInteger)ch * (NSUInteger)ch * sizeof(uint16_t);
        NSUInteger aBytes = (NSUInteger)ch * (NSUInteger)sp * sizeof(uint16_t);
        id<MTLBuffer> wBuf = [device newBufferWithLength:wBytes * (NSUInteger)depth options:MTLResourceStorageModeShared];
        id<MTLBuffer> aBuf = [device newBufferWithLength:aBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithLength:aBytes options:MTLResourceStorageModeShared];
        memset(wBuf.contents, 0x20, wBuf.length);
        memset(aBuf.contents, 0x3C, aBytes);

        MPSMatrix *left[512];
        if (depth > 512) {
            *wallMs = -1;
            *gpuMs = -1;
            return;
        }
        for (int i = 0; i < depth; i++)
            left[i] = [[MPSMatrix alloc] initWithBuffer:wBuf offset:wBytes * (NSUInteger)i descriptor:wDesc];
        MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
        MPSMatrix *B = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:aDesc];

        double wall[64], gpu[64];
        for (int iter = 0; iter < kWarm + kIters; iter++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            MPSMatrix *inM = A, *outM = B;
            for (int layer = 0; layer < depth; layer++) {
                [mul encodeToCommandBuffer:cb leftMatrix:left[layer] rightMatrix:inM resultMatrix:outM];
                MPSMatrix *swap = inM; inM = outM; outM = swap;
            }
            uint64_t t0 = mach_absolute_time();
            [cb commit];
            [cb waitUntilCompleted];
            if (iter >= kWarm) {
                wall[iter - kWarm] = ticksToMs(mach_absolute_time() - t0);
                gpu[iter - kWarm] = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
            }
        }
        *wallMs = median(wall, kIters);
        *gpuMs = median(gpu, kIters);
    }
}

static double benchCPU(int ch, int sp, int depth) {
    size_t mat = (size_t)ch * (size_t)ch;
    size_t vec = (size_t)ch * (size_t)sp;
    float *W = malloc(sizeof(float) * mat * (size_t)depth);
    float *A = malloc(sizeof(float) * vec);
    float *B = malloc(sizeof(float) * vec);
    if (!W || !A || !B) return -1;
    for (size_t i = 0; i < mat * (size_t)depth; i++)
        W[i] = ((arc4random() & 1023) / 1023.0f) * 0.001f;
    for (size_t i = 0; i < vec; i++)
        A[i] = 0.01f;

    double samples[64];
    for (int iter = 0; iter < kWarm + kIters; iter++) {
        uint64_t t0 = mach_absolute_time();
        float *in = A, *out = B;
        for (int layer = 0; layer < depth; layer++) {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        ch, sp, ch, 1.f, W + (size_t)layer * mat, ch, in, ch, 0.f, out, ch);
            float *tmp = in; in = out; out = tmp;
        }
        double ms = ticksToMs(mach_absolute_time() - t0);
        if (iter >= kWarm) samples[iter - kWarm] = ms;
        if (in[0] == 1e30f) printf(" ");
    }
    free(W); free(A); free(B);
    return median(samples, kIters);
}

static int checkAgreement(id<MTLDevice> device, MPSGraphDevice *mpsDevice, id<MTLCommandQueue> queue) {
    const int ch = 32, sp = 8, depth = 2;
    float *W = calloc((size_t)depth * ch * ch, sizeof(float));
    float *x = calloc((size_t)ch * sp, sizeof(float));
    float *y = calloc((size_t)ch * sp, sizeof(float));
    float *tmp = calloc((size_t)ch * sp, sizeof(float));
    for (int i = 0; i < depth * ch * ch; i++) W[i] = ((i % 7) - 3) * 0.02f;
    for (int i = 0; i < ch * sp; i++) x[i] = ((i % 5) - 2) * 0.1f;
    memcpy(y, x, sizeof(float) * (size_t)ch * sp);
    float *in = y, *out = tmp;
    for (int layer = 0; layer < depth; layer++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    ch, sp, ch, 1.f, W + layer * ch * ch, ch, in, sp, 0.f, out, sp);
        float *swap = in; in = out; out = swap;
    }
    float *ref = in;

    MPSGraph *graph = [MPSGraph new];
    graph.options = MPSGraphOptionsNone;
    MPSGraphTensor *inT = [graph placeholderWithShape:@[@(ch), @(sp)] dataType:MPSDataTypeFloat32 name:@"x"];
    MPSGraphTensor *cur = inT;
    for (int layer = 0; layer < depth; layer++) {
        NSData *wd = [NSData dataWithBytes:W + layer * ch * ch length:sizeof(float) * ch * ch];
        MPSGraphTensor *wt = [graph constantWithData:wd shape:@[@(ch), @(ch)] dataType:MPSDataTypeFloat32];
        cur = [graph matrixMultiplicationWithPrimaryTensor:wt secondaryTensor:cur name:nil];
    }
    MPSGraphShapedType *shaped = [[MPSGraphShapedType alloc] initWithShape:@[@(ch), @(sp)] dataType:MPSDataTypeFloat32];
    MPSGraphExecutable *exe = [graph compileWithDevice:mpsDevice feeds:@{inT: shaped}
                                         targetTensors:@[cur] targetOperations:nil compilationDescriptor:nil];
    id<MTLBuffer> buf = [device newBufferWithBytes:x length:sizeof(float) * ch * sp options:MTLResourceStorageModeShared];
    id<MTLBuffer> obuf = [device newBufferWithLength:sizeof(float) * ch * sp options:MTLResourceStorageModeShared];
    MPSGraphTensorData *inData = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:@[@(ch), @(sp)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *outData = [[MPSGraphTensorData alloc] initWithMTLBuffer:obuf shape:@[@(ch), @(sp)] dataType:MPSDataTypeFloat32];
    MPSGraphExecutableExecutionDescriptor *ed = [MPSGraphExecutableExecutionDescriptor new];
    ed.waitUntilCompleted = YES;
    exe.options = MPSGraphOptionsSynchronizeResults;
    [exe runWithMTLCommandQueue:queue inputsArray:@[inData] resultsArray:@[outData] executionDescriptor:ed];
    float *got = obuf.contents;
    float maxAbs = 0;
    for (int i = 0; i < ch * sp; i++) {
        float e = fabsf(got[i] - ref[i]);
        if (e > maxAbs) maxAbs = e;
    }
    printf("agreement 32x8 depth2 fp32 max abs error %.3e\n", maxAbs);
    free(W); free(x); free(y); free(tmp);
    return maxAbs < 1e-4f;
}

int main(void) {
    mach_timebase_info(&g_tb);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "no Metal device\n");
        return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MPSGraphDevice *mpsDevice = [MPSGraphDevice deviceWithMTLDevice:device];
    printf("device %s\n", device.name.UTF8String);
    printf("GPU graph: compiled MPSGraph, fp16, D GEMMs, median of %d after %d warmup\n", kIters, kWarm);
    printf("GPU mps: MPSMatrixMultiplication chain in one command buffer, fp16\n");
    printf("         wall = commit to completion; gpu = GPUEndTime-GPUStartTime\n");
    printf("CPU: D sequential cblas_sgemm, fp32, Accelerate on the CPU matrix unit\n");
    printf("FLOPs = 2*C*C*S*D\n\n");
    if (!checkAgreement(device, mpsDevice, queue)) {
        fprintf(stderr, "GPU and CPU disagree on the small fp32 chain\n");
        return 2;
    }

    Config cf[] = {
        {512, 64, 1},
        {512, 64, 32},
        {512, 64, 64},
        {512, 64, 128},
        {256, 64, 256},
        {1024, 64, 1},
        {2048, 64, 1},
        {512, 2048, 16},
        {2048, 2048, 4},
        {4096, 4096, 1},
    };
    int n = (int)(sizeof(cf) / sizeof(cf[0]));
    printf("%-22s %8s %8s %8s %8s %8s %8s %8s\n",
           "config", "GFLOP", "graph", "mpsWall", "mpsGPU", "mpsTF", "cpu", "cpuTF");
    printf("%-22s %8s %8s %8s %8s %8s %8s %8s\n",
           "", "", "TFLOPS", "ms", "ms", "TFLOPS", "ms", "TFLOPS");
    for (int i = 0; i < n; i++) {
        int c = cf[i].c, s = cf[i].s, d = cf[i].d;
        double gf = gflops(c, s, d);
        double graph = -1, wall = -1, gpu = -1, cpu = -1;
        @try {
            graph = benchGraph(device, mpsDevice, queue, c, s, d);
        } @catch (NSException *ex) {
            fprintf(stderr, "graph exception %s: %s\n", ex.name.UTF8String, ex.reason.UTF8String);
        }
        @try {
            benchMPS(device, queue, c, s, d, &wall, &gpu);
        } @catch (NSException *ex) {
            fprintf(stderr, "mps exception %s: %s\n", ex.name.UTF8String, ex.reason.UTF8String);
        }
        cpu = benchCPU(c, s, d);
        char label[64];
        snprintf(label, sizeof(label), "%dx %dch sp%d", d, c, s);
        printf("%-22s %8.2f %8.2f %8.3f %8.3f %8.2f %8.3f %8.2f\n",
               label, gf,
               graph > 0 ? gf / graph : 0,
               wall, gpu,
               gpu > 0 ? gf / gpu : 0,
               cpu, cpu > 0 ? gf / cpu : 0);
        fflush(stdout);
    }
    return 0;
}
