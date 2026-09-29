# /// script
# dependencies = ["opencv-python-headless", "matplotlib"]
# ///
"""Click the ball in sampled frames -> labels.csv, training data for a ball detector (TrackNet).

uv run label_ball.py 1912.mp4 1913.mp4 1914.mp4 [--runs 40 --run-len 10]

Left click: ball is here.  n or right click: ball not visible.  b: back one frame.  q: quit.
Every label is saved immediately; rerun the same command to resume where you stopped.
Zoom with the toolbar magnifier (the zoom is kept between frames); turn it off again before clicking.
"""
import argparse
import csv
import os
import random

import cv2
import matplotlib.pyplot as plt

OUT = "labels.csv"


def plan(videos, runs, run_len, seed=0):
    """Deterministic list of (video, frame): `runs` stretches of `run_len` consecutive frames.

    ponytail: uniform random, so many runs land between rallies; bias toward
    detector tracks if too few frames show the ball.
    """
    rng = random.Random(seed)
    counts = {v: int(cv2.VideoCapture(v).get(cv2.CAP_PROP_FRAME_COUNT)) for v in videos}
    items = []
    for _ in range(runs):
        v = rng.choice(videos)
        start = rng.randrange(counts[v] - run_len)
        items += [(v, f) for f in range(start, start + run_len)]
    return items


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        return {(r["video"], int(r["frame"])): (int(r["visible"]),
                float(r["x"]) if r["x"] else None, float(r["y"]) if r["y"] else None)
                for r in csv.DictReader(fh)}


def save(path, labels):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "frame", "visible", "x", "y"])
        for (v, f), (vis, x, y) in sorted(labels.items()):
            w.writerow([v, f, vis, "" if x is None else round(x, 1), "" if y is None else round(y, 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--run-len", type=int, default=10)
    a = ap.parse_args()

    items = plan(a.videos, a.runs, a.run_len)
    labels = load(OUT)
    caps = {v: cv2.VideoCapture(v) for v in a.videos}
    state = {"i": next((k for k, it in enumerate(items) if it not in labels), len(items))}

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.subplots_adjust(0, 0, 1, 0.95)
    ax.set_axis_off()
    img = ax.imshow([[0]])
    prev_mark, = ax.plot([], [], "c+", ms=14, mew=1)  # ball in previous frame, as a hint
    cur_mark, = ax.plot([], [], "rx", ms=10, mew=2)  # existing label for this frame

    def show():
        i = state["i"]
        if i >= len(items):
            print(f"done: {len(labels)} frames labelled in {OUT}")
            plt.close(fig)
            return
        v, f = items[i]
        caps[v].set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = caps[v].read()
        first = img.get_array().shape[:2] == (1, 1)
        img.set_data(frame[..., ::-1])
        if first:
            img.set_extent((-0.5, frame.shape[1] - 0.5, frame.shape[0] - 0.5, -0.5))
            ax.set_xlim(-0.5, frame.shape[1] - 0.5)
            ax.set_ylim(frame.shape[0] - 0.5, -0.5)
        for mark, key in ((prev_mark, (v, f - 1)), (cur_mark, (v, f))):
            lab = labels.get(key)
            mark.set_data(*(([lab[1]], [lab[2]]) if lab and lab[0] else ([], [])))
        done = sum(it in labels for it in items)
        ax.set_title(f"{done}/{len(items)} labelled   {v} frame {f}   "
                     "click = ball   n / right click = not visible   b = back   q = quit", fontsize=9)
        fig.canvas.draw_idle()

    def label(vis, x=None, y=None):
        labels[items[state["i"]]] = (vis, x, y)
        save(OUT, labels)
        state["i"] += 1
        show()

    def on_click(e):
        if e.inaxes is not ax or fig.canvas.toolbar.mode:  # ignore clicks while zooming/panning
            return
        if e.button == 1:
            label(1, e.xdata, e.ydata)
        elif e.button == 3:
            label(0)

    def on_key(e):
        if e.key == "n":
            label(0)
        elif e.key == "b":
            state["i"] = max(0, state["i"] - 1)
            show()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()


def _selfcheck():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "l.csv")
    labels = {("a.mp4", 3): (1, 12.3, 45.6), ("a.mp4", 4): (0, None, None)}
    save(path, labels)
    assert load(path) == labels, load(path)


if __name__ == "__main__":
    _selfcheck()
    main()
