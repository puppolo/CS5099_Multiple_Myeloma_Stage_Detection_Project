import sys
import datetime
import time
import pathlib


class _Log:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def setup_logging(out_dir: pathlib.Path) -> tuple:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    terminal = sys.stdout
    log_file = open(out_dir / f"run_{ts}.txt", "w", encoding="utf-8")
    sys.stdout = _Log(terminal, log_file)
    print(f"Log     : {out_dir}/run_{ts}.txt")
    print(f"Figures : {out_dir}/")
    return out_dir, time.time()


def setup_matplotlib(out_dir: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _counter = [0]

    def _save_show(*args, **kwargs):
        _counter[0] += 1
        fname = pathlib.Path(out_dir) / f"fig_{_counter[0]:02d}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"[figure saved: {fname.name}]")
        plt.close()

    plt.show = _save_show
    return plt
