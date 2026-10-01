#include "internal.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <stdio.h>
#import <string.h>

static double ms_since(uint64_t t0) {
    static mach_timebase_info_data_t tb;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ mach_timebase_info(&tb); });
    return (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
}

static Class gDesc, gModelClass, gRequest, gSurface;
static int gReady;

static int ane_init(void) {
    if (gReady)
        return gDesc ? 0 : -1;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    gDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    gModelClass = NSClassFromString(@"_ANEInMemoryModel");
    gRequest = NSClassFromString(@"_ANERequest");
    gSurface = NSClassFromString(@"_ANEIOSurfaceObject");
    gReady = 1;
    return (gDesc && gModelClass && gRequest && gSurface) ? 0 : -1;
}

static NSString *value_name(const AnerOp *ops, int id) {
    if (ops[id].kind == ANER_INPUT)
        return [NSString stringWithFormat:@"in%d_f16", ops[id].input_index];
    return [NSString stringWithFormat:@"v%d", id];
}

static int is_blob(AnerOpKind kind) {
    return kind == ANER_CONV || kind == ANER_CONST;
}

static int blob_count(const AnerGraph *graph) {
    int n = 0;
    for (int i = 0; i < graph->nops; i++)
        if (is_blob(graph->ops[i].kind))
            n++;
    return n;
}

static int payload_count(const AnerOp *op, int spatial) {
    if (op->kind == ANER_CONV)
        return op->channels * op->in_channels;
    if (op->kind == ANER_CONST)
        return op->channels * spatial;
    return 0;
}

static uint64_t hash_fp16(const float *w, int count) {
    uint64_t h = 14695981039346656037ull;
    for (int k = 0; k < count; k++) {
        __fp16 half = (__fp16)w[k];
        uint16_t bits;
        memcpy(&bits, &half, 2);
        h ^= bits;
        h *= 1099511628211ull;
    }
    return h;
}

static int same_fp16(const float *a, const float *b, int count) {
    for (int k = 0; k < count; k++) {
        __fp16 ha = (__fp16)a[k];
        __fp16 hb = (__fp16)b[k];
        uint16_t ba, bb;
        memcpy(&ba, &ha, 2);
        memcpy(&bb, &hb, 2);
        if (ba != bb)
            return 0;
    }
    return 1;
}

/* One weight.bin. The compiler rejects more than 16 separate weight files.
   A chunk with the size and absolute payload offset is what returns the right
   values. Giving every layer its own chunk streams the whole file and cuts a
   128-step chain from about 13 TFLOPS to about 7. Layers whose fp16 weights
   match share a chunk, which is the resident-weight case. */
static int same_blob(const AnerOp *a, const AnerOp *b) {
    if (a->kind != b->kind || a->channels != b->channels)
        return 0;
    if (a->kind == ANER_CONV) {
        if (a->in_channels != b->in_channels)
            return 0;
        return same_fp16(a->weight, b->weight, a->channels * a->in_channels);
    }
    if (a->kind == ANER_CONST)
        return same_fp16(a->weight, b->weight, a->channels);
    return 0;
}

static void plan_offsets(const AnerGraph *graph, uint64_t *offsets, int spatial) {
    int n = blob_count(graph);
    if (n == 0)
        return;
    uint64_t *hashes = calloc((size_t)n, sizeof(uint64_t));
    int *ownerOp = calloc((size_t)n, sizeof(int));
    int *ownerBlob = calloc((size_t)n, sizeof(int));
    int unique = 0;
    size_t cursor = 64;
    int blob = 0;
    for (int i = 0; i < graph->nops; i++) {
        if (!is_blob(graph->ops[i].kind))
            continue;
        const AnerOp *op = &graph->ops[i];
        int keyCount = op->kind == ANER_CONST ? op->channels : op->channels * op->in_channels;
        uint64_t hash = hash_fp16(op->weight, keyCount);
        int found = -1;
        for (int u = 0; u < unique; u++) {
            const AnerOp *prev = &graph->ops[ownerOp[u]];
            if (hashes[u] != hash || !same_blob(prev, op))
                continue;
            found = u;
            break;
        }
        if (found >= 0) {
            offsets[blob] = offsets[ownerBlob[found]];
        } else {
            offsets[blob] = cursor;
            hashes[unique] = hash;
            ownerOp[unique] = i;
            ownerBlob[unique] = blob;
            unique++;
            cursor += 64 + (size_t)payload_count(op, spatial) * 2;
        }
        blob++;
    }
    free(hashes);
    free(ownerOp);
    free(ownerBlob);
}

static void write_payload(uint16_t *fp, const AnerOp *op, int spatial) {
    if (op->kind == ANER_CONV) {
        int count = op->channels * op->in_channels;
        for (int k = 0; k < count; k++) {
            __fp16 h = (__fp16)op->weight[k];
            memcpy(fp + k, &h, sizeof(h));
        }
        return;
    }
    for (int c = 0; c < op->channels; c++) {
        __fp16 h = (__fp16)op->weight[c];
        for (int s = 0; s < spatial; s++)
            memcpy(fp + c * spatial + s, &h, sizeof(h));
    }
}

static NSData *weight_blob(const AnerGraph *graph, const uint64_t *offsets, int spatial) {
    int n = blob_count(graph);
    if (n == 0)
        return nil;
    size_t total = 64;
    int blob = 0;
    for (int i = 0; i < graph->nops; i++) {
        if (!is_blob(graph->ops[i].kind))
            continue;
        size_t bytes = 64 + (size_t)payload_count(&graph->ops[i], spatial) * 2;
        if (offsets[blob] + bytes > total)
            total = offsets[blob] + bytes;
        blob++;
    }
    uint8_t *buf = calloc(total, 1);
    if (!buf)
        return nil;
    buf[0] = 1;
    buf[4] = 2;
    blob = 0;
    for (int i = 0; i < graph->nops; i++) {
        if (!is_blob(graph->ops[i].kind))
            continue;
        int owner = 1;
        for (int p = 0; p < blob; p++)
            if (offsets[p] == offsets[blob])
                owner = 0;
        if (owner) {
            const AnerOp *op = &graph->ops[i];
            int count = payload_count(op, spatial);
            uint32_t ws = (uint32_t)count * 2;
            uint8_t *chunk = buf + offsets[blob];
            chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
            chunk[4] = 1;
            uint32_t dataOff = (uint32_t)(offsets[blob] + 64);
            memcpy(chunk + 8, &ws, sizeof(ws));
            memcpy(chunk + 16, &dataOff, sizeof(dataOff));
            write_payload((uint16_t *)(chunk + 64), op, spatial);
        }
        blob++;
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSString *mil_source(const AnerGraph *graph, int spatial) {
    NSMutableString *mil = [NSMutableString string];
    [mil appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n    func main<ios18>("];
    for (int i = 0, seen = 0; i < graph->nops; i++) {
        if (graph->ops[i].kind != ANER_INPUT)
            continue;
        if (seen)
            [mil appendString:@", "];
        [mil appendFormat:@"tensor<fp32, [1, %d, 1, %d]> in%d", graph->ops[i].channels, spatial, graph->ops[i].input_index];
        seen++;
    }
    [mil appendString:@") {\n"];
    [mil appendString:
        @"            string c_pad_type_0 = const()[name = string(\"c_pad_type_0\"), val = string(\"valid\")];\n"
        @"            tensor<int32, [2]> c_strides_0 = const()[name = string(\"c_strides_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            tensor<int32, [4]> c_pad_0 = const()[name = string(\"c_pad_0\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"            tensor<int32, [2]> c_dilations_0 = const()[name = string(\"c_dilations_0\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            int32 c_groups_0 = const()[name = string(\"c_groups_0\"), val = int32(1)];\n"
        @"            string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        @"            string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"];
    for (int i = 0; i < graph->nops; i++) {
        const AnerOp *op = &graph->ops[i];
        if (op->kind != ANER_INPUT)
            continue;
        [mil appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> in%d_f16 = cast(dtype = to_fp16, x = in%d)[name = string(\"cast_in_%d\")];\n",
            op->channels, spatial, op->input_index, op->input_index, op->input_index];
    }
    int nblob = blob_count(graph);
    uint64_t *offsets = nblob ? calloc((size_t)nblob, sizeof(uint64_t)) : NULL;
    plan_offsets(graph, offsets, spatial);
    int blobIndex = 0;
    for (int i = 0; i < graph->nops; i++) {
        const AnerOp *op = &graph->ops[i];
        if (op->kind == ANER_INPUT)
            continue;
        NSString *dst = [NSString stringWithFormat:@"v%d", i];
        if (op->kind == ANER_CONST) {
            [mil appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> %@ = const()[name = string(\"%@\"), val = tensor<fp16, [1, %d, 1, %d]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n",
                op->channels, spatial, dst, dst, op->channels, spatial, offsets[blobIndex]];
            blobIndex++;
        } else if (op->kind == ANER_CONV) {
            NSString *src = value_name(graph->ops, op->a);
            NSString *wname = [NSString stringWithFormat:@"W%d", blobIndex];
            [mil appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> %@ = const()[name = string(\"%@\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n",
                op->channels, op->in_channels, wname, wname, op->channels, op->in_channels, offsets[blobIndex]];
            [mil appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> %@ = conv(dilations = c_dilations_0, groups = c_groups_0, pad = c_pad_0, pad_type = c_pad_type_0, strides = c_strides_0, weight = %@, x = %@)[name = string(\"%@\")];\n",
                op->channels, spatial, dst, wname, src, dst];
            blobIndex++;
        } else if (op->kind == ANER_ADD) {
            [mil appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> %@ = add(x = %@, y = %@)[name = string(\"%@\")];\n",
                op->channels, spatial, dst, value_name(graph->ops, op->a), value_name(graph->ops, op->b), dst];
        } else if (op->kind == ANER_MUL) {
            [mil appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> %@ = mul(x = %@, y = %@)[name = string(\"%@\")];\n",
                op->channels, spatial, dst, value_name(graph->ops, op->a), value_name(graph->ops, op->b), dst];
        } else if (op->kind == ANER_SIGMOID) {
            [mil appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> %@ = sigmoid(x = %@)[name = string(\"%@\")];\n",
                op->channels, spatial, dst, value_name(graph->ops, op->a), dst];
        }
    }
    NSMutableArray<NSString *> *results = [NSMutableArray array];
    for (int slot = 0; slot < graph->noutputs; slot++) {
        for (int i = 0; i < graph->nops; i++) {
            if (!graph->ops[i].is_output || graph->ops[i].output_index != slot)
                continue;
            NSString *src = value_name(graph->ops, i);
            NSString *name = [NSString stringWithFormat:@"out%d", slot];
            [mil appendFormat:@"            tensor<fp32, [1, %d, 1, %d]> %@ = cast(dtype = to_fp32, x = %@)[name = string(\"%@\")];\n",
                graph->ops[i].channels, spatial, name, src, name];
            [results addObject:name];
        }
    }
    [mil appendFormat:@"        } -> (%@);\n}\n", [results componentsJoinedByString:@", "]];
    free(offsets);
    return mil;
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

@interface ANEREngineProgram : NSObject {
@public
    id model;
    id request;
    NSString *tmpDir;
    IOSurfaceRef *inputs;
    IOSurfaceRef *outputs;
    int *inChannels;
    int *outChannels;
    int ninputs;
    int noutputs;
    int spatial;
}
@end

@implementation ANEREngineProgram
- (void)dealloc {
    for (int i = 0; i < ninputs; i++)
        if (inputs[i]) CFRelease(inputs[i]);
    for (int i = 0; i < noutputs; i++)
        if (outputs[i]) CFRelease(outputs[i]);
    free(inputs);
    free(outputs);
    free(inChannels);
    free(outChannels);
    if (tmpDir)
        [[NSFileManager defaultManager] removeItemAtPath:tmpDir error:nil];
    NSError *err = nil;
    if (model)
        ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &err);
}
@end

static int shape_fits_neural_engine(const AnerGraph *graph) {
    for (int i = 0; i < graph->nops; i++) {
        int channels = graph->ops[i].channels;
        if (channels <= 0 || channels > 512 || (channels % 64) != 0)
            return 0;
        if (graph->ops[i].kind == ANER_CONV) {
            int in_channels = graph->ops[i].in_channels;
            if (in_channels <= 0 || in_channels > 512 || (in_channels % 64) != 0)
                return 0;
        }
    }
    return 1;
}

void *aner_ane_compile(const AnerGraph *graph, int spatial, double *compile_ms) {
    if (compile_ms)
        *compile_ms = 0;
    if (!shape_fits_neural_engine(graph) || ane_init() != 0)
        return NULL;
    @autoreleasepool {
        NSString *mil = mil_source(graph, spatial);
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        int nblob = blob_count(graph);
        uint64_t *offsets = nblob ? calloc((size_t)nblob, sizeof(uint64_t)) : NULL;
        plan_offsets(graph, offsets, spatial);
        NSData *blob = weight_blob(graph, offsets, spatial);
        free(offsets);
        NSMutableDictionary *weights = [NSMutableDictionary dictionary];
        if (blob)
            weights[@"@model_path/weights/weight.bin"] = @{@"offset": @0, @"data": blob};
        NSError *err = nil;
        id desc = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            gDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, weights, nil);
        if (!desc)
            return NULL;
        id model = ((id (*)(Class, SEL, id))objc_msgSend)(gModelClass, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model)
            return NULL;
        id hex = ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:hex];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmp stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmp stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        for (NSString *key in weights) {
            NSString *rel = [key stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
            [weights[key][@"data"] writeToFile:[tmp stringByAppendingPathComponent:rel] atomically:YES];
        }
        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &err);
        if (!ok) {
            fprintf(stderr, "ANE graph compile failed: %s\n", err.description.UTF8String);
            [fm removeItemAtPath:tmp error:nil];
            return NULL;
        }
        ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &err);
        if (compile_ms)
            *compile_ms = ms_since(t0);
        if (!ok) {
            fprintf(stderr, "ANE graph load failed: %s\n", err.localizedDescription.UTF8String);
            [fm removeItemAtPath:tmp error:nil];
            return NULL;
        }

        ANEREngineProgram *program = [ANEREngineProgram new];
        program->model = model;
        program->tmpDir = tmp;
        program->spatial = spatial;
        program->ninputs = graph->ninputs;
        program->noutputs = graph->noutputs;
        program->inputs = calloc((size_t)graph->ninputs, sizeof(IOSurfaceRef));
        program->outputs = calloc((size_t)graph->noutputs, sizeof(IOSurfaceRef));
        program->inChannels = calloc((size_t)graph->ninputs, sizeof(int));
        program->outChannels = calloc((size_t)graph->noutputs, sizeof(int));
        NSMutableArray *inWraps = [NSMutableArray array];
        NSMutableArray *inIdx = [NSMutableArray array];
        NSMutableArray *outWraps = [NSMutableArray array];
        NSMutableArray *outIdx = [NSMutableArray array];
        for (int i = 0; i < graph->nops; i++) {
            if (graph->ops[i].kind == ANER_INPUT)
                program->inChannels[graph->ops[i].input_index] = graph->ops[i].channels;
            if (graph->ops[i].is_output)
                program->outChannels[graph->ops[i].output_index] = graph->ops[i].channels;
        }
        for (int i = 0; i < program->ninputs; i++) {
            NSUInteger bytes = (NSUInteger)program->inChannels[i] * (NSUInteger)spatial * sizeof(float);
            program->inputs[i] = make_surface(bytes);
            id wrap = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(gSurface, @selector(objectWithIOSurface:), program->inputs[i]);
            [inWraps addObject:wrap];
            [inIdx addObject:@(i)];
        }
        for (int i = 0; i < program->noutputs; i++) {
            NSUInteger bytes = (NSUInteger)program->outChannels[i] * (NSUInteger)spatial * sizeof(float);
            program->outputs[i] = make_surface(bytes);
            id wrap = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(gSurface, @selector(objectWithIOSurface:), program->outputs[i]);
            [outWraps addObject:wrap];
            [outIdx addObject:@(i)];
        }
        program->request = ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            gRequest,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            inWraps, inIdx, outWraps, outIdx, nil, nil, @0);
        return (void *)CFBridgingRetain(program);
    }
}

int aner_ane_eval(void *handle, const float *const *inputs, float *const *outputs, double *exec_ms) {
    ANEREngineProgram *program = (__bridge ANEREngineProgram *)handle;
    if (!program)
        return -1;
    for (int i = 0; i < program->ninputs; i++) {
        NSUInteger bytes = (NSUInteger)program->inChannels[i] * (NSUInteger)program->spatial * sizeof(float);
        IOSurfaceLock(program->inputs[i], 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(program->inputs[i]), inputs[i], bytes);
        IOSurfaceUnlock(program->inputs[i], 0, NULL);
    }
    NSError *err = nil;
    uint64_t t0 = mach_absolute_time();
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        program->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, program->request, &err);
    if (exec_ms)
        *exec_ms = ms_since(t0);
    if (!ok) {
        fprintf(stderr, "ANE graph eval failed: %s\n", err.localizedDescription.UTF8String);
        return -1;
    }
    for (int i = 0; i < program->noutputs; i++) {
        NSUInteger bytes = (NSUInteger)program->outChannels[i] * (NSUInteger)program->spatial * sizeof(float);
        IOSurfaceLock(program->outputs[i], kIOSurfaceLockReadOnly, NULL);
        memcpy(outputs[i], IOSurfaceGetBaseAddress(program->outputs[i]), bytes);
        IOSurfaceUnlock(program->outputs[i], kIOSurfaceLockReadOnly, NULL);
    }
    return 0;
}

void aner_ane_release(void *handle) {
    if (handle)
        CFRelease(handle);
}
