import csv
import time
from contextlib import contextmanager
from collections import defaultdict

from configs.v2v_config import TIMINGS_CSV_FILE


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

    def init_timings_file(self, base_dir):
        timings_file = base_dir / TIMINGS_CSV_FILE

        if not timings_file.exists():
            with timings_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "exp_name",
                        "frames",
                        "t_total",
                        "t_easi3r",
                        "t_ddim",
                        "t_dust3r",
                        "t_diffusion",
                        "t_metrics",
                        "t_visualize",
                        "t_4dgs",
                        "t_upsample",
                    ]
                )

        return timings_file


    def finish_timings(self, base_dir, exp_name, n_frames):

        timings_file = self.init_timings_file(base_dir)
        times = self.as_dict()

        with timings_file.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                exp_name,
                n_frames,
                f"{times.get('total', 0.0):.3f}",
                f"{times.get('easi3r', 0.0):.3f}",
                f"{times.get('ddim', 0.0):.3f}",
                f"{times.get('dust3r', 0.0):.3f}",
                f"{times.get('diffusion', 0.0):.3f}",
                f"{times.get('metrics', 0.0):.3f}",
                f"{times.get('visualize', 0.0):.3f}",
                f"{times.get('4dgs', 0.0):.3f}",
                f"{times.get('upsample', 0.0):.3f}",
            ])