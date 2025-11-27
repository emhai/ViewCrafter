import time
from contextlib import contextmanager
from collections import defaultdict

class RunTimer:
    def __init__(self):
        self.timings = defaultdict(float)  # name -> total seconds

    @contextmanager
    def time(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.timings[name] += (end - start)

    def reset(self):
        self.timings.clear()

    def as_dict(self):
        return dict(self.timings)