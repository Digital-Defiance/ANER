# ANER

ANER is the Apple Neural Engine Runtime. It evaluates static numerical graphs on the Apple Neural Engine. A value is a matrix with one column per ensemble member. Channels are the state. Coefficients are compiled into the graph, and a repeated linear map stays resident across the horizon.

The public header is `lib/include/aner.h`. Version 0.1.0. `lib/include/aner_lut6.h` compiles a frozen combinational LUT6 netlist into one of those graphs. HELUT calls it for a frozen cleartext circuit of at most 64 lookups and a batch of at most 64, with the wire vector inside 512 channels. Wider batches, larger circuits, flip-flops, and HELUT's encrypted modular path stay on Metal.

## Requirements

macOS on Apple Silicon, with the Xcode command-line tools (`clang`). The runtime loads the Neural Engine from the private AppleNeuralEngine framework. No extra packages are required.

## Build, test, install

```sh
make -C lib
make -C lib test
make -C lib install PREFIX=/usr/local
```

`make test` runs the numerical checks and the 512-mode forecast. Install writes:

| Path | File |
| --- | --- |
| `include/aner.h` | public header |
| `include/aner_lut6.h` | LUT6 netlist compiler |
| `lib/libaner.dylib` | shared library (`libaner.0.dylib`) |
| `lib/libaner.a` | static archive |
| `lib/pkgconfig/aner.pc` | pkg-config |
| `bin/aner-modal` | the ensemble example |

`PREFIX` defaults to `/usr/local`. `DESTDIR` is supported. `make -C lib uninstall PREFIX=...` removes those files.

Link a program with the shared library:

```sh
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
clang forecast.c $(pkg-config --cflags --libs aner) -o forecast
```

The archive is linked by path. `--static --libs-only-other` supplies the frameworks:

```sh
clang forecast.c $(pkg-config --cflags aner) \
  "$(pkg-config --variable=libdir aner)/libaner.a" \
  $(pkg-config --static --libs-only-other aner) -o forecast
```

## Model

Each tensor is `[channels, spatial]`, row-major, fp32 at the boundary. Spatial is fixed when the graph is compiled. `aner_run_batch` cuts a wider ensemble into tiles of that width and zero-pads a short final tile. Pointwise ops and 1×1 maps treat columns independently.

| Call | Role |
| --- | --- |
| `aner_input` | a field replaced on every run |
| `aner_const` | one coefficient per channel, copied into every column |
| `aner_conv` | row-major linear map |
| `aner_scale` | diagonal map |
| `aner_repeat` | the same square map, applied for many steps |
| `aner_add`, `aner_mul` | elementwise |
| `aner_sigmoid` | logistic |
| `aner_horner` | elementwise power series, low coefficient first |
| `aner_gelu` | `x * sigmoid(1.702 * x)` |

`aner_compile` copies the graph. The caller may free the graph afterward. `aner_uses_neural_engine` reports which backend is live. `aner_run_reference` always runs the float32 interpreter.

The Neural Engine path requires every channel count to be a multiple of 64 and at most 512. Other shapes use the interpreter. On an M4 Max, a repeated 512×512 map at spatial 64 runs at about 13 TFLOPS once the chain is 128 steps; giving every step its own matrix drops that to about 7 TFLOPS because the weights no longer stay resident. fp16 is the on-chip format. A 128-step modal rollout differs from the float32 probe by a few hundredths on states of order 1.

## Running on both engines

`aner_compile_scheduled` picks the compute engines:

| Schedule | Backend |
| --- | --- |
| `ANER_SCHED_ANE` | Neural Engine, then the interpreter. The default, and what `aner_compile` selects. |
| `ANER_SCHED_GPU` | Metal, then the interpreter. |
| `ANER_SCHED_SPLIT` | Neural Engine and GPU at once. |

`aner_run_batch` tiles a wide batch across whichever engines the schedule enables, handing each the next chunk of columns as it falls idle. The Neural Engine takes 64-column tiles; the GPU takes a wide chunk in one dispatch, because Metal is far more efficient on wide inputs than on a single tile. Both GPU schedules use that wide chunk: `ANER_SCHED_GPU` feeds the whole batch to the GPU in wide dispatches, and `ANER_SCHED_SPLIT` runs the Neural Engine and GPU at once, the faster engine naturally taking more of the batch so the run finishes near the busier engine's time. A single tile (`aner_run`, or a batch equal to the compiled spatial width) runs on one engine. If one backend is unavailable the other carries the whole batch; if neither is, the interpreter does. `aner_uses_gpu` reports whether the Metal backend is live.

The GPU runs the same op set as the Neural Engine (input, const, conv, scale, repeat, add, mul, sigmoid, and the pointwise helpers built from them), lowered to one MPSGraph executable at optimization level 0. Level 0 is deliberate: the default level 1 runs a placement pass that moves fp16 matmul chains onto the Neural Engine, which would put both halves of a split on the same engine. Both backends compute in fp16, so a split batch matches an Neural-Engine-only batch to fp16 rounding, not to the float32 interpreter.

GPU tuning (environment variables, read when a program is compiled or run; for benchmarking):

| Variable | Default | Effect |
| --- | --- | --- |
| `ANER_GPU_CHUNK` | 4096 | Target GPU chunk width in columns (rounded down to a multiple of the tile). |
| `ANER_GPU_PIPELINE` | 1 | `0` uses the synchronous GPU worker; `1` keeps two chunks in flight, packing the next straight into shared buffers while the GPU computes. |
| `ANER_GPU_OPT_LEVEL` | 0 | `1` compiles at MPSGraph optimization level 1, whose placement pass may move fp16 matmuls onto the Neural Engine. Experimental. |

Measured on an M4 Max with HALO's scorer (512-wide residual MLP, 18 linear layers, 65,536 columns): GPU-only rose from 1.08M to 1.67M columns/s and the split from 1.47M to 2.03M, mostly from pipelining (2048: 1.51M / 1.86M) and the 4096 chunk. MLX fp16 on the same network: 1.49M. Optimization level 1 gave no gain and at 8192 moved work onto the Neural Engine (split fell to 1.43M), so level 0 stays the default.

The GPU lowering keeps consts as `[channels, 1]` columns that broadcast for free, and lowers a diagonal square conv (`aner_scale`, the scale inside `aner_gelu`) to an elementwise multiply instead of a dense matmul.

On an M4 Max, the 512-mode, 128-step, 4096-column rollout runs at about 12 TFLOPS on the Neural Engine alone, about 11 on the GPU alone, and about 23 split across both, close to 1.9× over either engine. The split gain needs a batch wide enough to keep both engines fed; a batch that fits in one tile runs on a single engine.

## Example

Thirty-two steps of one map, 64 modes, 128 initial conditions, tiles of 64:

```c
#include <aner.h>
#include <stdlib.h>

int main(void) {
    const int n = 64, steps = 32, batch = 128;
    float *A = calloc((size_t)n * n, sizeof(float));
    float *X = calloc((size_t)n * batch, sizeof(float));
    float *Y = calloc((size_t)n * batch, sizeof(float));
    for (int i = 0; i < n; i++)
        A[i * n + i] = 0.99f;

    AnerGraph *g = aner_graph_new();
    int x = aner_input(g, n);
    int y = aner_repeat(g, x, steps, A);
    aner_output(g, y);
    AnerProgram *p = aner_compile(g, 64);
    aner_graph_free(g);

    const float *inputs[] = {X};
    float *outputs[] = {Y};
    double ms = 0;
    aner_run_batch(p, inputs, outputs, batch, &ms);

    aner_program_free(p);
    free(A);
    free(X);
    free(Y);
    return 0;
}
```

## Python

Numerical code uses the `aner` package. An array is NumPy `float32` with shape `(channels, columns)`, one column per ensemble member. Installing the package builds `libaner` and ships that library inside the package.

```sh
pip install .
aner-modal
```

`aner-modal` is the 512-mode damped ensemble: 4096 initial conditions, 128 steps, a sensor row, and amplitude percentiles. `python -m aner.modal` is the same program. During development, `make -C lib python-test` runs the wrapper checks and that study against `lib/build`.

```python
import numpy as np
import aner

n, steps, batch = 64, 32, 128
A = np.zeros((n, n), np.float32)
np.fill_diagonal(A, 0.99)
X = np.zeros((n, batch), np.float32)

graph = aner.Graph()
state = graph.input(n)
graph.output(graph.repeat(state, steps, A))
program = graph.compile(64)
Y = program.run_batch(X)
```

`Y` has shape `(n, batch)`. One output comes back as an array. Several outputs come back as a list, in the order they were marked. `program.uses_neural_engine` and `program.uses_gpu` report the live backends, and `program.eval_ms` is the last evaluation. `graph.reference(spatial, X)` runs the float32 interpreter. Float64 arrays are converted to float32.

`graph.compile(spatial, aner.Schedule.SPLIT)` runs a wide batch on the Neural Engine and the GPU together. `aner.Schedule.GPU` uses Metal alone, and `aner.Schedule.ANE` (the default) uses the Neural Engine.

```python
program = graph.compile(64, aner.Schedule.SPLIT)
Y = program.run_batch(X)          # 4096 columns spread across both engines
assert program.uses_neural_engine and program.uses_gpu
```

`lib/examples/modal.c` is a 512-mode damped oscillator: 4096 initial conditions, 128 steps, then a sensor row and amplitude percentiles. After install it is `aner-modal`. From the build tree it is `lib/build/modal`.

```sh
aner-modal [--scenarios N] [--horizon D] [--modes C]
```

Modes must be even, a multiple of 64, and at most 512.
