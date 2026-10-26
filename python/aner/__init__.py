"""NumPy interface to the Apple Neural Engine Runtime.

A value is a float32 array of shape (channels, columns). Each column is one
ensemble member. The library is the macOS build of libaner.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

__all__ = ["Graph", "Program", "Schedule", "Tensor", "version"]


class Schedule:
    """Which compute engine(s) a program's batch runs on.

    ANE    Neural Engine, then the interpreter. The default.
    GPU    Metal, then the interpreter.
    SPLIT  Neural Engine and GPU at once; a wide batch is handed to whichever
           engine is free, so both run in parallel.
    """

    ANE = 0
    GPU = 1
    SPLIT = 2


class _Graph(ctypes.Structure):
    pass


class _Program(ctypes.Structure):
    pass


_FloatP = ctypes.POINTER(ctypes.c_float)


def _library_path() -> str:
    override = os.environ.get("ANER_LIBRARY")
    if override:
        return override
    here = Path(__file__).resolve().parent
    vendored = here / "libaner.0.dylib"
    if vendored.is_file():
        return str(vendored)
    dev = here.parents[1] / "lib" / "build" / "libaner.dylib"
    if dev.is_file():
        return str(dev)
    try:
        libdir = subprocess.check_output(
            ["pkg-config", "--variable=libdir", "aner"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        libdir = ""
    if libdir:
        installed = Path(libdir) / "libaner.dylib"
        if installed.is_file():
            return str(installed)
    raise OSError(
        "libaner was not found. Install this package with pip, or set ANER_LIBRARY."
    )


def _load():
    lib = ctypes.CDLL(_library_path())
    g = ctypes.POINTER(_Graph)
    p = ctypes.POINTER(_Program)
    fp = _FloatP
    fpp = ctypes.POINTER(fp)
    lib.aner_version.restype = ctypes.c_char_p
    lib.aner_graph_new.restype = g
    lib.aner_graph_free.argtypes = [g]
    lib.aner_input.argtypes = [g, ctypes.c_int]
    lib.aner_input.restype = ctypes.c_int
    lib.aner_const.argtypes = [g, ctypes.c_int, fp]
    lib.aner_const.restype = ctypes.c_int
    lib.aner_conv.argtypes = [g, ctypes.c_int, ctypes.c_int, ctypes.c_int, fp]
    lib.aner_conv.restype = ctypes.c_int
    lib.aner_scale.argtypes = [g, ctypes.c_int, fp]
    lib.aner_scale.restype = ctypes.c_int
    lib.aner_repeat.argtypes = [g, ctypes.c_int, ctypes.c_int, fp]
    lib.aner_repeat.restype = ctypes.c_int
    lib.aner_add.argtypes = [g, ctypes.c_int, ctypes.c_int]
    lib.aner_add.restype = ctypes.c_int
    lib.aner_mul.argtypes = [g, ctypes.c_int, ctypes.c_int]
    lib.aner_mul.restype = ctypes.c_int
    lib.aner_sigmoid.argtypes = [g, ctypes.c_int]
    lib.aner_sigmoid.restype = ctypes.c_int
    lib.aner_horner.argtypes = [g, ctypes.c_int, fp, ctypes.c_int]
    lib.aner_horner.restype = ctypes.c_int
    lib.aner_gelu.argtypes = [g, ctypes.c_int]
    lib.aner_gelu.restype = ctypes.c_int
    lib.aner_output.argtypes = [g, ctypes.c_int]
    lib.aner_output.restype = ctypes.c_int
    lib.aner_compile.argtypes = [g, ctypes.c_int]
    lib.aner_compile.restype = p
    lib.aner_compile_scheduled.argtypes = [g, ctypes.c_int, ctypes.c_int]
    lib.aner_compile_scheduled.restype = p
    lib.aner_program_free.argtypes = [p]
    lib.aner_uses_neural_engine.argtypes = [p]
    lib.aner_uses_neural_engine.restype = ctypes.c_int
    lib.aner_uses_gpu.argtypes = [p]
    lib.aner_uses_gpu.restype = ctypes.c_int
    lib.aner_compile_milliseconds.argtypes = [p]
    lib.aner_compile_milliseconds.restype = ctypes.c_double
    lib.aner_run.argtypes = [p, fpp, fpp, ctypes.POINTER(ctypes.c_double)]
    lib.aner_run.restype = ctypes.c_int
    lib.aner_run_reference.argtypes = [g, ctypes.c_int, fpp, fpp]
    lib.aner_run_reference.restype = ctypes.c_int
    lib.aner_run_batch.argtypes = [p, fpp, fpp, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    lib.aner_run_batch.restype = ctypes.c_int
    return lib


_lib = _load()


def version() -> str:
    raw = _lib.aner_version()
    return raw.decode() if raw else ""


def _check(code: int, what: str) -> int:
    if code < 0:
        raise RuntimeError(what)
    return code


def _vector(values, size: int, what: str) -> np.ndarray:
    data = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if data.size != size:
        raise ValueError(f"{what} has length {data.size}, expected {size}")
    return data


def _matrix(values, rows: int, cols: int, what: str) -> np.ndarray:
    data = np.ascontiguousarray(values, dtype=np.float32)
    if data.shape != (rows, cols):
        raise ValueError(f"{what} has shape {data.shape}, expected {(rows, cols)}")
    return data


def _as_inputs(values):
    if isinstance(values, np.ndarray):
        return [values]
    return list(values)


def _pointers(arrays: list[np.ndarray]):
    ptrs = (_FloatP * len(arrays))()
    for i, array in enumerate(arrays):
        ptrs[i] = array.ctypes.data_as(_FloatP)
    return ptrs


def _prepare(arrays, channels: list[int], columns: int) -> list[np.ndarray]:
    if len(arrays) != len(channels):
        raise ValueError(f"expected {len(channels)} inputs, got {len(arrays)}")
    prepared = []
    for array, count in zip(arrays, channels):
        data = np.ascontiguousarray(array, dtype=np.float32)
        if data.shape != (count, columns):
            raise ValueError(f"input has shape {data.shape}, expected {(count, columns)}")
        prepared.append(data)
    return prepared


class Tensor:
    def __init__(self, graph: Graph, ident: int, channels: int):
        self.graph = graph
        self.id = ident
        self.channels = channels

    def __repr__(self) -> str:
        return f"Tensor(channels={self.channels})"


class Program:
    def __init__(self, ptr, spatial: int, input_channels: list[int], output_channels: list[int]):
        self._ptr = ptr
        self.spatial = spatial
        self.input_channels = list(input_channels)
        self.output_channels = list(output_channels)
        self.eval_ms = 0.0

    def close(self) -> None:
        if self._ptr:
            _lib.aner_program_free(self._ptr)
            self._ptr = None

    def __enter__(self) -> Program:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _require(self):
        if not self._ptr:
            raise RuntimeError("program is closed")

    @property
    def uses_neural_engine(self) -> bool:
        self._require()
        return bool(_lib.aner_uses_neural_engine(self._ptr))

    @property
    def uses_gpu(self) -> bool:
        self._require()
        return bool(_lib.aner_uses_gpu(self._ptr))

    @property
    def compile_ms(self) -> float:
        self._require()
        return float(_lib.aner_compile_milliseconds(self._ptr))

    def run(self, inputs):
        self._require()
        prepared = _prepare(_as_inputs(inputs), self.input_channels, self.spatial)
        return self._invoke(_lib.aner_run, prepared, self.spatial, None)

    def run_batch(self, inputs):
        self._require()
        arrays = _as_inputs(inputs)
        if not arrays:
            raise ValueError("expected at least one input")
        batch = int(np.ascontiguousarray(arrays[0], dtype=np.float32).shape[-1])
        prepared = _prepare(arrays, self.input_channels, batch)
        return self._invoke(_lib.aner_run_batch, prepared, batch, batch)

    def _invoke(self, fn, prepared: list[np.ndarray], columns: int, batch):
        outputs = [np.empty((count, columns), dtype=np.float32) for count in self.output_channels]
        ms = ctypes.c_double()
        in_ptrs = _pointers(prepared)
        out_ptrs = _pointers(outputs)
        if batch is None:
            rc = fn(self._ptr, in_ptrs, out_ptrs, ctypes.byref(ms))
        else:
            rc = fn(self._ptr, in_ptrs, out_ptrs, batch, ctypes.byref(ms))
        if rc != 0:
            raise RuntimeError("aner run failed")
        self.eval_ms = float(ms.value)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def __repr__(self) -> str:
        if not self._ptr:
            where = "closed"
        else:
            engines = []
            if self.uses_neural_engine:
                engines.append("Neural Engine")
            if self.uses_gpu:
                engines.append("GPU")
            where = " + ".join(engines) if engines else "CPU"
        return f"Program({where}, spatial={self.spatial})"


class Graph:
    def __init__(self):
        ptr = _lib.aner_graph_new()
        if not ptr:
            raise RuntimeError("aner_graph_new failed")
        self._ptr = ptr
        self._input_channels: list[int] = []
        self._output_channels: list[int] = []

    def close(self) -> None:
        if self._ptr:
            _lib.aner_graph_free(self._ptr)
            self._ptr = None

    def __enter__(self) -> Graph:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _require(self):
        if not self._ptr:
            raise RuntimeError("graph is closed")

    def _tensor(self, ident: int, channels: int, what: str) -> Tensor:
        _check(ident, what)
        return Tensor(self, ident, channels)

    def _id(self, tensor: Tensor) -> int:
        self._require()
        if not isinstance(tensor, Tensor) or tensor.graph is not self:
            raise TypeError("expected a tensor from this graph")
        return tensor.id

    def input(self, channels: int) -> Tensor:
        self._require()
        if channels <= 0:
            raise ValueError("channels must be positive")
        ident = _lib.aner_input(self._ptr, int(channels))
        self._input_channels.append(int(channels))
        return self._tensor(ident, int(channels), "aner_input failed")

    def const(self, values) -> Tensor:
        self._require()
        data = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        if data.size <= 0:
            raise ValueError("const needs one value per channel")
        ident = _lib.aner_const(self._ptr, int(data.size), data.ctypes.data_as(_FloatP))
        return self._tensor(ident, int(data.size), "aner_const failed")

    def conv(self, tensor: Tensor, weight) -> Tensor:
        ident = self._id(tensor)
        data = np.ascontiguousarray(weight, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != tensor.channels:
            raise ValueError(f"weight has shape {data.shape}, expected (out, {tensor.channels})")
        out = _lib.aner_conv(
            self._ptr, ident, int(data.shape[0]), int(data.shape[1]), data.ctypes.data_as(_FloatP)
        )
        return self._tensor(out, int(data.shape[0]), "aner_conv failed")

    def scale(self, tensor: Tensor, scales) -> Tensor:
        data = _vector(scales, tensor.channels, "scales")
        ident = _lib.aner_scale(self._ptr, self._id(tensor), data.ctypes.data_as(_FloatP))
        return self._tensor(ident, tensor.channels, "aner_scale failed")

    def repeat(self, tensor: Tensor, steps: int, matrix) -> Tensor:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        data = _matrix(matrix, tensor.channels, tensor.channels, "matrix")
        ident = _lib.aner_repeat(
            self._ptr, self._id(tensor), int(steps), data.ctypes.data_as(_FloatP)
        )
        return self._tensor(ident, tensor.channels, "aner_repeat failed")

    def add(self, a: Tensor, b: Tensor) -> Tensor:
        if a.channels != b.channels:
            raise ValueError("add tensors have different channel counts")
        ident = _lib.aner_add(self._ptr, self._id(a), self._id(b))
        return self._tensor(ident, a.channels, "aner_add failed")

    def mul(self, a: Tensor, b: Tensor) -> Tensor:
        if a.channels != b.channels:
            raise ValueError("mul tensors have different channel counts")
        ident = _lib.aner_mul(self._ptr, self._id(a), self._id(b))
        return self._tensor(ident, a.channels, "aner_mul failed")

    def sigmoid(self, tensor: Tensor) -> Tensor:
        ident = _lib.aner_sigmoid(self._ptr, self._id(tensor))
        return self._tensor(ident, tensor.channels, "aner_sigmoid failed")

    def horner(self, tensor: Tensor, coeffs) -> Tensor:
        data = np.ascontiguousarray(coeffs, dtype=np.float32).reshape(-1)
        if data.size < 1:
            raise ValueError("horner needs at least one coefficient")
        ident = _lib.aner_horner(self._ptr, self._id(tensor), data.ctypes.data_as(_FloatP), int(data.size))
        return self._tensor(ident, tensor.channels, "aner_horner failed")

    def gelu(self, tensor: Tensor) -> Tensor:
        ident = _lib.aner_gelu(self._ptr, self._id(tensor))
        return self._tensor(ident, tensor.channels, "aner_gelu failed")

    def output(self, tensor: Tensor) -> int:
        ident = self._id(tensor)
        index = _check(_lib.aner_output(self._ptr, ident), "aner_output failed")
        while len(self._output_channels) <= index:
            self._output_channels.append(0)
        self._output_channels[index] = tensor.channels
        return index

    def compile(self, spatial: int, schedule: int = Schedule.ANE) -> Program:
        self._require()
        if spatial <= 0:
            raise ValueError("spatial must be positive")
        if schedule not in (Schedule.ANE, Schedule.GPU, Schedule.SPLIT):
            raise ValueError("schedule must be a Schedule value")
        if not self._output_channels:
            raise RuntimeError("mark at least one output")
        ptr = _lib.aner_compile_scheduled(self._ptr, int(spatial), int(schedule))
        if not ptr:
            raise RuntimeError("aner_compile failed")
        return Program(ptr, int(spatial), self._input_channels, self._output_channels)

    def reference(self, spatial: int, inputs):
        self._require()
        if not self._output_channels:
            raise RuntimeError("mark at least one output")
        prepared = _prepare(_as_inputs(inputs), self._input_channels, int(spatial))
        outputs = [np.empty((count, int(spatial)), dtype=np.float32) for count in self._output_channels]
        rc = _lib.aner_run_reference(
            self._ptr, int(spatial), _pointers(prepared), _pointers(outputs)
        )
        if rc != 0:
            raise RuntimeError("aner_run_reference failed")
        if len(outputs) == 1:
            return outputs[0]
        return outputs
