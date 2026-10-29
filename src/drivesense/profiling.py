from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

import numpy as np


class StageProfiler:
    """Measure execution time for individual pipeline stages."""

    def __init__(self):
        self.timings = defaultdict(list)

    @contextmanager
    def measure(self, stage_name: str):
        start = perf_counter()

        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.timings[stage_name].append(elapsed)

    def summary(self):
        """
        Return structured timing statistics for each stage.

        All reported durations are measured in milliseconds.
        """
        results = {}

        for name, values in self.timings.items():
            if not values:
                continue

            durations_ms = np.asarray(values, dtype=float) * 1000

            results[name] = {
                "calls": int(len(durations_ms)),
                "average_ms": float(np.mean(durations_ms)),
                "median_ms": float(np.median(durations_ms)),
                "p95_ms": float(np.percentile(durations_ms, 95)),
                "total_seconds": float(np.sum(durations_ms) / 1000),
            }

        return results

    def report(self):
        print("\n========== STAGE PROFILING REPORT ==========")

        total_stage_time = sum(
            sum(values) for values in self.timings.values()
        )

        for name, values in sorted(
            self.timings.items(),
            key=lambda item: sum(item[1]),
            reverse=True,
        ):
            if not values:
                continue

            total = sum(values)
            average_ms = (total / len(values)) * 1000

            percentage = (
                total / total_stage_time * 100
                if total_stage_time > 0
                else 0
            )

            p95_ms = float(
                np.percentile(np.asarray(values) * 1000, 95)
            )

            print(
                f"{name:20} | "
                f"calls={len(values):4d} | "
                f"avg={average_ms:8.2f} ms | "
                f"p95={p95_ms:8.2f} ms | "
                f"total={total:7.2f} s | "
                f"share={percentage:5.1f}%"
            )
