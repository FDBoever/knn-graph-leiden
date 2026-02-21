import time
import os
import psutil
import pandas as pd


class PipelineTracker:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.records = []
        self.process = psutil.Process(os.getpid())

    def start(self, stage):
        self._stage = stage
        self._t0 = time.perf_counter()
        self._cpu0 = self.process.cpu_percent(interval=None)
        self._mem0 = self.process.memory_info().rss

    def stop(self):
        t1 = time.perf_counter()
        cpu1 = self.process.cpu_percent(interval=None)
        mem1 = self.process.memory_info().rss

        record = {
            "stage": self._stage,
            "seconds": round(t1 - self._t0, 4),
            "cpu_percent_delta": round(cpu1, 2),
            "memory_delta_mb": round((mem1 - self._mem0) / 1024**2, 3),
            "memory_current_mb": round(mem1 / 1024**2, 3),
        }

        self.records.append(record)

        #if self.verbose:
        #    print(f"     {record['stage']}: "
        #          f"{record['seconds']}s | "
        #          f"Δmem={record['memory_delta_mb']}MB | "
        #          f"mem={record['memory_current_mb']}MB")

    def save(self, path):
        df = pd.DataFrame(self.records)
        df.to_csv(path, sep="\t", index=False)

    def summary(self):
        return pd.DataFrame(self.records)