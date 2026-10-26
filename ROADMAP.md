# Making ANER a dependable Apple Silicon compute runtime

2026-10-26. This is a proposal based on the HALO integration measurements, not a
list of features already implemented. The production runtime remains unchanged.

ANER already has the important foundation: a small static graph API, resident
weights for repeated maps, ANE and Metal implementations, an FP32 CPU reference,
and concurrent scheduling across the ANE and GPU. The next step is to make its
placement, precision, and performance predictable for applications.

## Priority relative to the next HALO run

**None of the broader work below is a prerequisite for resuming HALO.** The
pipeline changes, batched feature encoder, explicit ANE tile width, and MLX FP32
option are implemented and tested. The fixed replay was approximately 43%
faster than the previous pipeline, with identical classifications. Further
inference tuning had little effect on whole-replay latency.

Resume the search first. Use its new labels to evaluate frontier Hc>=3/4 yield,
improve the scorer, and guide the forced-chain construction work. Generalizing
ANER should not consume that research budget without a measured HALO bottleneck.

| Trigger | First response | Broader ANER work needed first? |
| --- | --- | --- |
| The GPU is free | Use the tested MLX FP32 scorer | No |
| GPU training competes with scoring | Select existing ANE inference at width 256 | No |
| An accelerator unexpectedly falls back to CPU | Add/inspect execution diagnostics and strict fallback handling | Yes, before relying on that ANER path for a long run |
| Packing or allocation dominates a real workload | Profile and add buffer reuse/layout improvements | Only after measuring that cost |
| An application requires accelerated FP32 through ANER's API | Productize the GPU FP32 prototype with precision and fallback tests | Yes for that application; HALO already has MLX FP32 |
| Search remains on the Hc=4 plateau | Improve proposals, labels/scoring, and designed constructions | No runtime feature is known to resolve this |

For broader ANER development, prioritize precision contracts and truthful
execution reporting, then measured automatic selection and buffer reuse.
Operator expansion and portability backends follow concrete consumer needs.

## First: precision and truthful execution reports

1. **Make precision explicit in the compile API.** Keep existing calls compatible,
   and add compile options for FP16 and FP32. FP32 must use a genuine FP32 GPU or
   CPU path; it must not quietly hand some columns to the FP16 ANE. An FP32 GPU
   prototype exists in the HALO benchmark scratch directory and passed the
   scorer's reference comparison. It is not a release-ready API.
2. **Report actual execution, including fallback.** Return engine/chunk counts,
   fallback counts and reasons, packing time, device time, and total wall time.
   In `program.c`, a failed chunk can currently recover through
   `aner_cpu_run` while the program's compiled-engine flag remains set. Those
   flags describe capability, not reliable evidence that a particular batch ran
   entirely on that engine. Applications need a strict mode that fails rather
   than silently moving a performance-critical workload to the CPU.
3. **Test the numerical contract.** Extend the existing reference tests with
   real models, small/large/tail batches, deep repeated maps, poorly conditioned
   inputs, and engine failures. Report max/RMS error and application metrics:
   HALO's averaged Hc error and top-k agreement, forecast errors and percentiles,
   or exact output agreement for discrete circuits. FP16 on two engines does
   not imply identical rounding; split scheduling can change which engine handles
   a column. Offer fixed placement for applications needing repeatability, and
   document tolerances rather than promising universal bitwise reproducibility.

Apple's [typed execution documentation](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html)
excludes the Neural Engine from FP32-typed ML programs, while allowing FP32 on
the GPU and CPU. The current private ANE lowering also explicitly uses FP16.
An ANER FP32 feature should therefore start with Metal and the existing CPU
reference; emulating FP32 on the ANE is a separate research project.

## Second: choose configurations using whole-call latency

4. **Add constrained automatic selection.** The current split scheduler already
   estimates chunk costs. Extend that to choose ANE tile width, GPU chunk width,
   and ANE/GPU/split placement for a graph, precision, and expected batch-size
   range. Keep explicit overrides and a policy to reserve the GPU for training
   or other work. Bound calibration/compilation time and cache the result by
   graph/weights, precision, hardware, OS/runtime version, and batch-size range.
   Use an explicit application-controlled cache directory.
5. **Reduce repeated host-side work.** Cache immutable I/O shapes, reuse worker
   scratch and buffers, and avoid clearing full tiles before overwriting them.
   Consider caller-provided input/output buffers and an asynchronous submission
   interface for streaming workloads. Measure Python-to-runtime conversion,
   packing, copies, allocation, and synchronization alongside device execution.
   GPU double buffering is already implemented; extend reuse rather than
   rebuilding that mechanism.
6. **Profile graph lowering and layout.** Compare fusion, matrix layouts, chunk
   sizes, and compilation optimization levels on representative workloads.
   Benchmark FP32 and FP16 separately. Consider an optional existing backend
   adapter where it wins, rather than assuming every graph should use ANER's
   current MPSGraph lowering or writing custom kernels before measuring.

## Then expand coverage where real projects need it

7. Add operations such as reductions, reshape/transpose, gather, and broader
   channel dimensions in response to concrete applications. Preserve the
   independent-column contract of the current batch scheduler; operations that
   mix samples require explicit grouping and scheduling semantics.
8. Keep a small benchmark suite spanning repeated linear systems, residual
   MLPs, streaming batches, parameter ensembles, and LUT circuits. Record cold
   compile, warm latency, throughput, memory, numerical error, and fallback.
   Report energy or power only after measuring it. A single peak-TFLOPS case
   cannot establish the best backend for unrelated workloads.
9. Evaluate a supported Core ML backend as a portability/maintenance option,
   while retaining the private ANE path for experiments where it has measured
   value. Treat OS compatibility and backend diagnostics as part of the product.

## What HALO needs immediately

- Keep SAT and geometric certification exact on the CPU. ANER accelerates the
  learned ranking stage; it does not replace the combinatorial checker.
- Use batched occupancy feature construction. The measured replay spent more
  time creating features than running inference. HALO's shared scorer path now
  implements it, so both ANER and MLX benefit.
- Expose ANE tile width. HALO's MLP batch improved with width 256 versus the
  historical width 64; repeated-map workloads may prefer a different setting.
- Allow MLX FP32 when the GPU is free. In the current v4 batch, it was faster
  than ANER's initial FP32 GPU prototype and much closer to the FP32 reference
  than the ANE path. Keep ANE available when GPU contention makes it useful.
- Compare actual Hc>=3/4 discoveries per wall-hour. Model quality, proposal
  diversity, retry policy, and escaping the current plateau matter more to the
  mathematical goal than optimizing an already small inference cost.

For HALO, finish and measure these small pipeline changes, then resume searching.
The broader ANER API work can proceed as a separate, bounded engineering effort.
