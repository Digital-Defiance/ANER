# ANER

ANER is the Apple Neural Engine Runtime. It evaluates static numerical graphs on the Apple Neural Engine. A value is a matrix with one column per ensemble member. Channels are the state. Coefficients are compiled into the graph, and a repeated linear map stays resident across the horizon.

The public header is `lib/include/aner.h`. Version 0.1.0.

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

`Y` has shape `(n, batch)`. One output comes back as an array. Several outputs come back as a list, in the order they were marked. `program.uses_neural_engine` reports the live backend, and `program.eval_ms` is the last evaluation. `graph.reference(spatial, X)` runs the float32 interpreter. Float64 arrays are converted to float32.

`lib/examples/modal.c` is a 512-mode damped oscillator: 4096 initial conditions, 128 steps, then a sensor row and amplitude percentiles. After install it is `aner-modal`. From the build tree it is `lib/build/modal`.

```sh
aner-modal [--scenarios N] [--horizon D] [--modes C]
```

Modes must be even, a multiple of 64, and at most 512.
