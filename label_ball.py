# /// script
# dependencies = ["opencv-python-headless", "matplotlib"]
# ///
"""Click the ball in sampled frames -> labels.csv, training data for a ball detector (TrackNet).

uv run label_ball.py 1912.mp4 1913.mp4 1914.mp4 [--runs 40 --run-len 10]
uv run label_ball.py 1902.mp4 --from-run runs/model_1902   # frames where the model lost the ball
uv run label_ball.py 1936.mp4 --false-positives runs/v2_1936   # detections that are probably not the ball

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


def hard_frames(video, run_dir, per_gap=3, max_gap=45):
    """Frames from a ball_speed.py run where the model likely failed, as (video, frame).

    - first frames after each track ends when the next one starts within max_gap frames
      (ball lost mid-rally),
    - frames around suspicious bounce pairs (too short, too fast, or too slow).
    """
    spans = {}
    with open(os.path.join(run_dir, "track.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            f = int(r["frame"])
            s, e = spans.get(r["track"], (f, f))
            spans[r["track"]] = (min(s, f), max(e, f))
    spans = sorted(spans.values())
    frames = []
    for (_, end), (start, _) in zip(spans, spans[1:]):
        if 1 < start - end <= max_gap:
            frames += range(end + 1, min(end + 1 + per_gap, start))
    fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
    with open(os.path.join(run_dir, "speeds.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if float(r["dist_m"]) < 1 or float(r["speed_kmh"]) > 25 or float(r["dt_s"]) > 1:
                f0, f1 = round(float(r["t_s"]) * fps), round((float(r["t_s"]) + float(r["dt_s"])) * fps)
                frames += range(f0 - 2, f0 + 3)  # the first bounce
                frames += range((f0 + f1) // 2 - 2, (f0 + f1) // 2 + 3)  # a missed bounce would be mid-way
    return [(video, f) for f in sorted(set(frames)) if f >= 0]


def suspect_detections(video, run_dir, max_frames=150, spacing=5, borderline=0.7, near_px=20):
    """Frames where the model probably saw something that is NOT the ball -> (items, guesses, counts).

    Needs detections.csv and rejected.csv from a ball_speed.py --model run. Groups, sampled in turn:
    - untracked: detections the tracker didn't keep (lone hits, too-short tracks, static clutter),
    - borderline: tracked detections the model was barely sure of (score <= borderline),
    - rejected: bounce candidates the physics checks threw out (often one bad detection).
    Picked frames are at least `spacing` apart so the set isn't near-duplicates.
    guesses maps (video, frame) -> the model's (x, y), shown so a right guess costs one key.
    """
    def rows(name):
        with open(os.path.join(run_dir, name), newline="") as fh:
            return list(csv.DictReader(fh))

    tracked = {}
    for r in rows("track.csv"):
        tracked.setdefault(int(r["frame"]), []).append((float(r["x_px"]), float(r["y_px"])))

    def is_tracked(f, x, y):
        return any(abs(x - tx) <= near_px and abs(y - ty) <= near_px for tx, ty in tracked.get(f, []))

    groups = {"untracked": [], "borderline": [], "rejected": []}
    for r in rows("detections.csv"):
        f, x, y, s = int(r["frame"]), float(r["x_px"]), float(r["y_px"]), float(r["score"])
        if not is_tracked(f, x, y):
            groups["untracked"].append((f, x, y))
        elif s <= borderline:
            groups["borderline"].append((f, x, y))
    groups["rejected"] = [(int(r["frame"]), float(r["x_px"]), float(r["y_px"])) for r in rows("rejected.csv")]

    rng = random.Random(0)
    for g in groups.values():
        rng.shuffle(g)
    picked, counts = {}, dict.fromkeys(groups, 0)
    queues = {k: iter(v) for k, v in groups.items()}
    while len(picked) < max_frames and queues:
        for k in list(queues):
            for f, x, y in queues[k]:
                if all(abs(f - p) >= spacing for p in picked):
                    picked[f] = (x, y)
                    counts[k] += 1
                    break
            else:
                del queues[k]  # group used up
            if len(picked) >= max_frames:
                break
    items = [(video, f) for f in sorted(picked)]
    return items, {(video, f): picked[f] for f in picked}, counts


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
    ap.add_argument("--from-run", help="ball_speed.py output folder: label where the model struggled (one video only)")
    ap.add_argument("--false-positives", metavar="RUN_DIR",
                    help="ball_speed.py --model output folder: check detections that are probably not the ball")
    ap.add_argument("--max-frames", type=int, default=150, help="with --false-positives")
    a = ap.parse_args()

    guesses = {}
    if a.from_run or a.false_positives:
        assert len(a.videos) == 1, "--from-run / --false-positives take the one video that run was made from"
    if a.from_run:
        items = hard_frames(a.videos[0], a.from_run)
    elif a.false_positives:
        items, guesses, counts = suspect_detections(a.videos[0], a.false_positives, a.max_frames)
        print(f"{len(items)} frames to check: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
    else:
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
    guess_mark, = ax.plot([], [], "o", ms=22, mew=2, mfc="none", mec="yellow")  # model's guess (--false-positives)

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
        g = guesses.get((v, f))
        guess_mark.set_data(*(([g[0]], [g[1]]) if g else ([], [])))
        done = sum(it in labels for it in items)
        keys = ("yellow circle = model's guess   Enter = guess is the ball   n = NOT the ball / no ball   "
                "click = ball is elsewhere" if g else "click = ball   n / right click = not visible")
        ax.set_title(f"{done}/{len(items)} labelled   {v} frame {f}   {keys}   b = back   q = quit", fontsize=9)
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
        elif e.key == "enter" and items[state["i"]] in guesses:
            label(1, *guesses[items[state["i"]]])
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
    # Two tracks with a 6-frame gap -> the first 3 lost frames; a far-apart pair of tracks -> nothing.
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "track.csv"), "w") as fh:
        fh.write("track,frame,t_s,x_px,y_px\n0,10,0,0,0\n0,20,0,0,0\n1,27,0,0,0\n1,30,0,0,0\n2,500,0,0,0\n")
    with open(os.path.join(d, "speeds.csv"), "w") as fh:
        fh.write("track,t_s,from_m,to_m,dist_m,dt_s,speed_kmh\n")
    assert hard_frames("nonexistent.mp4", d) == [("nonexistent.mp4", f) for f in (21, 22, 23)]
    # Suspect detections: a sure tracked hit is skipped; a barely-sure tracked hit, a lone untracked hit
    # and a rejected bounce candidate are each picked once; a second lone hit 2 frames later is too close.
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "track.csv"), "w") as fh:
        fh.write("track,frame,t_s,x_px,y_px\n" + "".join(f"0,{f},0,100,100\n" for f in range(10, 15)))
    with open(os.path.join(d, "detections.csv"), "w") as fh:
        fh.write("frame,x_px,y_px,score\n10,100,100,0.95\n12,101,99,0.6\n50,500,300,0.9\n52,505,300,0.9\n")
    with open(os.path.join(d, "rejected.csv"), "w") as fh:
        fh.write("frame,x_px,y_px,reason\n30,200,200,paddle hit\n")
    items, guesses, counts = suspect_detections("v.mp4", d)
    assert [f for _, f in items] == [12, 30, 50], items
    assert guesses[("v.mp4", 50)] == (500.0, 300.0) and counts == {"untracked": 1, "borderline": 1, "rejected": 1}, (guesses, counts)


if __name__ == "__main__":
    _selfcheck()
    main()
