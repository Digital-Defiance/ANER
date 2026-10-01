"""512-mode damped ensemble, stepped with one resident linear map.

Each column is an initial condition. The sensor's first row is the fundamental
position. The amplitude ratios compare the stepped ensemble with the scalar
damping of each 2×2 block.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

import aner


def mode_rho(mode: int, nmodes: int) -> float:
    t = mode / (nmodes - 1)
    omega = 0.08 * (1.0 + t)
    zeta = 0.01 + 0.08 * t
    return math.exp(-zeta * omega)


def scenario_amp(scenario: int) -> float:
    return 0.2 + 1.3 * ((scenario * 13 + 7) % 100) / 100.0


def transition(modes: int) -> np.ndarray:
    nmodes = modes // 2
    matrix = np.zeros((modes, modes), np.float32)
    for i in range(nmodes):
        t = i / (nmodes - 1)
        omega = 0.08 * (1.0 + t)
        zeta = 0.01 + 0.08 * t
        wd = omega * math.sqrt(1.0 - zeta * zeta)
        rho = math.exp(-zeta * omega)
        c = np.float32(rho * math.cos(wd))
        s = np.float32(rho * math.sin(wd))
        r0 = 2 * i
        matrix[r0, r0] = c
        matrix[r0, r0 + 1] = -s
        matrix[r0 + 1, r0] = s
        matrix[r0 + 1, r0 + 1] = c
    return matrix


def initial_state(modes: int, scenarios: int) -> np.ndarray:
    nmodes = modes // 2
    state = np.zeros((modes, scenarios), np.float32)
    for s in range(scenarios):
        amp = np.float32(scenario_amp(s))
        for i in range(nmodes):
            phase = np.float32(0.2) * np.float32((s + 3 * i) % 13)
            state[2 * i, s] = amp * np.cos(phase)
            state[2 * i + 1, s] = amp * np.sin(phase)
    return state


def sensor_matrix(modes: int) -> np.ndarray:
    observed = np.arange(2, modes + 1, dtype=np.float32)[:, None]
    state = np.arange(1, modes + 1, dtype=np.float32)[None, :]
    matrix = np.zeros((modes, modes), np.float32)
    matrix[0, 0] = 1.0
    matrix[1:] = np.float32(0.02) * np.cos(np.float32(0.05) * observed * state)
    return matrix


def _percentile(values: np.ndarray, p: float) -> float:
    ordered = np.sort(values)
    idx = p * (len(ordered) - 1)
    i = int(idx)
    if i >= len(ordered) - 1:
        return float(ordered[-1])
    frac = idx - i
    return float(ordered[i] * (1.0 - frac) + ordered[i + 1] * frac)


class Forecast:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def forecast(modes: int = 512, scenarios: int = 4096, horizon: int = 128, tile: int = 64) -> Forecast:
    if modes < 2 or modes % 2 or modes > 512 or modes % 64 or not 1 <= scenarios <= 65536 or horizon < 1:
        raise ValueError("modes must be even, a multiple of 64, and at most 512")
    matrix = transition(modes)
    state0 = initial_state(modes, scenarios)
    observe = sensor_matrix(modes)

    step = aner.Graph()
    sense = aner.Graph()
    roll = sensor = None
    try:
        step.output(step.repeat(step.input(modes), horizon, matrix))
        sense.output(sense.conv(sense.input(modes), observe))
        roll = step.compile(tile)
        sensor = sense.compile(tile)
        started = time.perf_counter()
        stepped = roll.run_batch(state0)
        wall_ms = (time.perf_counter() - started) * 1e3
        sensed = sensor.run_batch(stepped)
        width = min(scenarios, tile)
        probe_started = time.perf_counter()
        probe = step.reference(width, state0[:, :width])
        probe_ms = (time.perf_counter() - probe_started) * 1e3
        probe_error = float(np.max(np.abs(stepped[:, :width] - probe)))
        sensor_error = float(np.max(np.abs(sensed[0] - stepped[0])))

        nmodes = modes // 2
        hi = nmodes - 1
        scale0 = mode_rho(0, nmodes) ** horizon
        scale_hi = mode_rho(hi, nmodes) ** horizon
        amp = np.array([scenario_amp(s) for s in range(scenarios)])
        measured0 = np.hypot(stepped[0], stepped[1])
        measured_hi = np.hypot(stepped[2 * hi], stepped[2 * hi + 1])
        fundamental = float(measured0.mean())
        highest = float(measured_hi.mean())
        predicted0 = float((amp * scale0).mean())
        predicted_hi = float((amp * scale_hi).mean())
        return Forecast(
            modes=modes,
            scenarios=scenarios,
            horizon=horizon,
            tile=tile,
            roll_on_neural_engine=roll.uses_neural_engine,
            sensor_on_neural_engine=sensor.uses_neural_engine,
            compile_ms=roll.compile_ms,
            eval_ms=roll.eval_ms,
            wall_ms=wall_ms,
            sensor_eval_ms=sensor.eval_ms,
            probe_ms=probe_ms,
            probe_error=probe_error,
            sensor_error=sensor_error,
            p05=_percentile(sensed[0], 0.05),
            p50=_percentile(sensed[0], 0.50),
            p95=_percentile(sensed[0], 0.95),
            fundamental_measured=fundamental,
            fundamental_predicted=predicted0,
            fundamental_ratio=fundamental / predicted0 if predicted0 else 0.0,
            highest_measured=highest,
            highest_predicted=predicted_hi,
            highest_ratio=highest / predicted_hi if predicted_hi else 0.0,
            fundamental_retained=100.0 * scale0,
            highest_retained=100.0 * scale_hi,
        )
    finally:
        if roll:
            roll.close()
        if sensor:
            sensor.close()
        step.close()
        sense.close()


def report(result: Forecast) -> str:
    roll = "Neural Engine" if result.roll_on_neural_engine else "CPU"
    sense = "Neural Engine" if result.sensor_on_neural_engine else "CPU"
    return "\n".join([
        "Modal forecast",
        f"  {result.modes} modes, {result.scenarios} scenarios, {result.horizon} steps, tile {result.tile}",
        f"  rollout {roll}, compile {result.compile_ms:.1f} ms, eval {result.eval_ms:.1f} ms, wall {result.wall_ms:.1f} ms",
        f"  float32 probe of first tile: {result.probe_ms:.1f} ms, max abs error {result.probe_error:.3e}",
        f"  sensor {sense}, eval {result.sensor_eval_ms:.1f} ms, fundamental error {result.sensor_error:.3e}",
        f"Fundamental position after {result.horizon} steps, across {result.scenarios} initial amplitudes",
        f"  p05 {result.p05:.4f}   p50 {result.p50:.4f}   p95 {result.p95:.4f}",
        "Modal amplitude, mean measured / undamped-step prediction",
        f"  fundamental  measured {result.fundamental_measured:.4f}  predicted {result.fundamental_predicted:.4f}  ratio {result.fundamental_ratio:.4f}",
        f"  highest mode measured {result.highest_measured:.4f}  predicted {result.highest_predicted:.4f}  ratio {result.highest_ratio:.4f}",
        f"  damping retained  fundamental {result.fundamental_retained:.0f}%  highest mode {result.highest_retained:.0f}%",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="512-mode damped ensemble on the Neural Engine")
    parser.add_argument("--scenarios", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--modes", type=int, default=512)
    args = parser.parse_args(argv)
    try:
        result = forecast(args.modes, args.scenarios, args.horizon)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(report(result))
    if not result.roll_on_neural_engine or result.probe_error > 5e-2:
        return 4
    return 0


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
