#ifndef ANER_LUT6_H
#define ANER_LUT6_H

#include <stdint.h>

/* A frozen LUT6 netlist, evaluated as one static graph.

   Wires are a vector of floats. Each column is an independent copy of the
   circuit, in the same sense as every other ANER program. inputs holds six
   wire indices per LUT, with -1 where that pin is absent (it reads as 0).
   entries holds 64 floats per LUT: entry k is the coefficient of the
   multilinear term whose bits select the six inputs. output wire indices are
   separate. level_luts lists LUT indices in evaluation order, and
   level_counts says how many of those indices belong to each combinational
   level. LUTs in one level all read the wires from the previous level.

   The whole wire vector has to fit in one state tensor, so wires must be in
   1...512. The compiled program pads the state up to a multiple of 64 and
   runs on the Neural Engine when that lowering succeeds. A sequential element
   is not part of this graph: clock the result with the host that owns the
   flip-flops.

   aner_lut6_forward reads and writes wires as batch-major, index
   batch_index * wires + wire. */

typedef struct AnerLUT6Program AnerLUT6Program;

AnerLUT6Program *aner_lut6_compile(
    int32_t wires,
    const int32_t *inputs,
    const int32_t *outputs,
    const float *entries,
    int32_t nlut,
    const int32_t *level_luts,
    const int32_t *level_counts,
    int32_t nlevels);

void aner_lut6_free(AnerLUT6Program *program);
int aner_lut6_wires(const AnerLUT6Program *program);
int aner_lut6_uses_neural_engine(const AnerLUT6Program *program);
double aner_lut6_compile_milliseconds(const AnerLUT6Program *program);

int aner_lut6_forward(AnerLUT6Program *program, float *wires, int32_t batch, double *exec_ms);

#endif
