#import "engines.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static double ms_since(uint64_t t0) {
    static mach_timebase_info_data_t tb;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ mach_timebase_info(&tb); });
    return (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
}

static Class gDesc, gModel, gRequest, gSurface;
static int gReady;

static int ane_init(void) {
    if (gReady)
        return gDesc ? 0 : -1;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    gDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    gModel = NSClassFromString(@"_ANEInMemoryModel");
    gRequest = NSClassFromString(@"_ANERequest");
    gSurface = NSClassFromString(@"_ANEIOSurfaceObject");
    gReady = 1;
    if (!gDesc || !gModel || !gRequest || !gSurface) {
        fprintf(stderr, "Neural Engine private classes unavailable\n");
        return -1;
    }
    return 0;
}

static NSString *mil_source(int ch, int sp, int depth) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ch, sp];
    [m appendString:@"            string c_pad_type_0 = const()[name = string(\"c_pad_type_0\"), val = string(\"valid\")];\n"
                    @"            tensor<int32, [2]> c_strides_0 = const()[name = string(\"c_strides_0\"), val = tensor<int32, [2]>([1, 1])];\n"
                    @"            tensor<int32, [4]> c_pad_0 = const()[name = string(\"c_pad_0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
                    @"            tensor<int32, [2]> c_dilations_0 = const()[name = string(\"c_dilations_0\"), val = tensor<int32, [2]>([1, 1])];\n"
                    @"            int32 c_groups_0 = const()[name = string(\"c_groups_0\"), val = int32(1)];\n"
                    @"            string x_to_fp16_dtype_0 = const()[name = string(\"x_to_fp16_dtype_0\"), val = string(\"fp16\")];\n"];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> x_to_fp16 = cast(dtype = x_to_fp16_dtype_0, x = x)[name = string(\"cast_in\")];\n", ch, sp];
    NSString *prev = @"x_to_fp16";
    for (int i = 0; i < depth; i++) {
        [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> W%d = const()[name = string(\"W%d\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
            ch, ch, i, i, ch, ch];
        NSString *out = [NSString stringWithFormat:@"c%d", i];
        [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> %@ = conv(dilations = c_dilations_0, groups = c_groups_0, pad = c_pad_0, pad_type = c_pad_type_0, strides = c_strides_0, weight = W%d, x = %@)[name = string(\"%@\")];\n",
            ch, sp, out, i, prev, out];
        prev = out;
    }
    [m appendString:@"            string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"];
    [m appendFormat:@"            tensor<fp32, [1, %d, 1, %d]> c = cast(dtype = to_fp32, x = %@)[name = string(\"cast_out\")];\n", ch, sp, prev];
    [m appendString:@"        } -> (c);\n}\n"];
    return m;
}

static NSData *weight_blob(const float *A, int ch) {
    int count = ch * ch;
    int ws = count * 2;
    int total = 128 + ws;
    uint8_t *buf = calloc((size_t)total, 1);
    if (!buf)
        return nil;
    buf[0] = 1;
    buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 1;
    uint32_t size = (uint32_t)ws;
    uint32_t dataOff = 128;
    memcpy(buf + 72, &size, sizeof(size));
    memcpy(buf + 80, &dataOff, sizeof(dataOff));
    uint16_t *fp = (uint16_t *)(buf + 128);
    for (int k = 0; k < count; k++) {
        __fp16 h = (__fp16)A[k];
        memcpy(fp + k, &h, sizeof(h));
    }
    return [NSData dataWithBytesNoCopy:buf length:(NSUInteger)total freeWhenDone:YES];
}

static IOSurfaceRef make_surface(NSUInteger bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

@interface ANEProgram : NSObject
@property (nonatomic, strong) id model;
@property (nonatomic) IOSurfaceRef inSurf;
@property (nonatomic) IOSurfaceRef outSurf;
@property (nonatomic, strong) id request;
@property (nonatomic, strong) NSString *tmpDir;
@property (nonatomic) int channels;
@property (nonatomic) int spatial;
@property (nonatomic) int depth;
@property (nonatomic) const float *matrix;
@end

@implementation ANEProgram
- (void)dealloc {
    if (_inSurf) CFRelease(_inSurf);
    if (_outSurf) CFRelease(_outSurf);
    if (_tmpDir)
        [[NSFileManager defaultManager] removeItemAtPath:_tmpDir error:nil];
}
@end

static NSMutableArray<ANEProgram *> *gPrograms;

static ANEProgram *program_for(const float *A, int ch, int sp, int depth, double *compile_ms, int *error) {
    *error = 0;
    if (compile_ms)
        *compile_ms = 0;
    if (ane_init() != 0) {
        *error = -1;
        return nil;
    }
    if (!gPrograms)
        gPrograms = [NSMutableArray array];
    for (ANEProgram *p in gPrograms) {
        if (p.matrix == A && p.channels == ch && p.spatial == sp && p.depth == depth)
            return p;
    }

    @autoreleasepool {
        NSError *err = nil;
        NSData *milData = [[mil_source(ch, sp, depth) dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSData *wb = weight_blob(A, ch);
        if (!milData || !wb) {
            *error = -2;
            return nil;
        }
        id desc = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            gDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData,
            @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}},
            nil);
        if (!desc) {
            *error = -3;
            return nil;
        }
        id model = ((id (*)(Class, SEL, id))objc_msgSend)(gModel, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) {
            *error = -4;
            return nil;
        }
        id hex = ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmp stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmp stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[tmp stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &err);
        if (!ok) {
            fprintf(stderr, "ANE compile failed: %s\n", err.localizedDescription.UTF8String);
            [fm removeItemAtPath:tmp error:nil];
            *error = -5;
            return nil;
        }
        ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &err);
        if (compile_ms)
            *compile_ms = ms_since(t0);
        if (!ok) {
            fprintf(stderr, "ANE load failed: %s\n", err.localizedDescription.UTF8String);
            [fm removeItemAtPath:tmp error:nil];
            *error = -6;
            return nil;
        }

        NSUInteger bytes = (NSUInteger)ch * (NSUInteger)sp * sizeof(float);
        IOSurfaceRef inSurf = make_surface(bytes);
        IOSurfaceRef outSurf = make_surface(bytes);
        if (!inSurf || !outSurf) {
            *error = -7;
            return nil;
        }
        id wIn = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(gSurface, @selector(objectWithIOSurface:), inSurf);
        id wOut = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(gSurface, @selector(objectWithIOSurface:), outSurf);
        id req = ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            gRequest,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        ANEProgram *program = [ANEProgram new];
        program.model = model;
        program.inSurf = inSurf;
        program.outSurf = outSurf;
        program.request = req;
        program.tmpDir = tmp;
        program.channels = ch;
        program.spatial = sp;
        program.depth = depth;
        program.matrix = A;
        [gPrograms addObject:program];
        return program;
    }
}

int ane_chain(const float *A, int channels, int depth, const float *X, float *Y, int spatial,
              double *compile_ms, double *exec_ms) {
    int error = 0;
    ANEProgram *program = program_for(A, channels, spatial, depth, compile_ms, &error);
    if (!program)
        return error ? error : -1;
    NSUInteger bytes = (NSUInteger)channels * (NSUInteger)spatial * sizeof(float);
    IOSurfaceLock(program.inSurf, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(program.inSurf), X, bytes);
    IOSurfaceUnlock(program.inSurf, 0, NULL);

    NSError *err = nil;
    uint64_t t0 = mach_absolute_time();
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        program.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, program.request, &err);
    if (exec_ms)
        *exec_ms = ms_since(t0);
    if (!ok) {
        fprintf(stderr, "ANE eval failed: %s\n", err.localizedDescription.UTF8String);
        return -8;
    }
    IOSurfaceLock(program.outSurf, kIOSurfaceLockReadOnly, NULL);
    memcpy(Y, IOSurfaceGetBaseAddress(program.outSurf), bytes);
    IOSurfaceUnlock(program.outSurf, kIOSurfaceLockReadOnly, NULL);
    return 0;
}

void ane_shutdown(void) {
    if (!gPrograms)
        return;
    for (ANEProgram *program in gPrograms) {
        NSError *err = nil;
        ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            program.model, @selector(unloadWithQoS:error:), 21, &err);
    }
    [gPrograms removeAllObjects];
}
