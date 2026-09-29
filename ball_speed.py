# /// script
# dependencies = ["opencv-python-headless", "numpy", "torch"]
# ///
"""Ping-pong ball speed from 1912.mp4: detect -> track -> table homography -> bounce-to-bounce speed.

uv run ball_speed.py 1912.mp4 [--debug out.mp4] [--model ball_net.pt]

Without --model the ball is found by colour + motion; with it, by the net from train_ball.py.
"""
import argparse
import csv

import cv2
import numpy as np

# Clicked on frame t=45s at 1280x720. Re-click them if the camera moves.
# Order: left end near, left end far, right end far, right end near (as seen in the image).
TABLE_PX = np.float32([[248.7, 292.9], [454.5, 221.6], [965.1, 348.8], [824.5, 494.1]])
TABLE_LEN_M, TABLE_WID_M = 2.8, 1.3
TABLE_M = np.float32([[0, TABLE_WID_M], [0, 0], [TABLE_LEN_M, 0], [TABLE_LEN_M, TABLE_WID_M]])

# Orange ball in HSV (OpenCV hue 0-180). Calibration knob: tune if detections are sparse or noisy.
HSV_LO, HSV_HI = (5, 120, 150), (22, 255, 255)
MIN_AREA, MAX_AREA = 3, 150  # px, ball is tiny at 720p (bigger when blurred)
MAX_JUMP_PX = 90  # max movement per frame to link a detection to the track
MAX_GAP = 4  # frames a track can go missing before it's dropped


def detect(frame, fg):
    """Moving orange blobs -> list of (x, y, area)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI) & fg
    mask = cv2.dilate(mask, None)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    return [(*cents[i], stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MIN_AREA <= stats[i, cv2.CC_STAT_AREA] <= MAX_AREA]


def read_frames(cap):
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


def color_detections(frames):
    bg = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False)
    for frame in frames:
        yield frame, detect(frame, bg.apply(frame))


def triplets(frames):
    """Yield (previous, current, next) for every frame; the ends repeat the edge frame, as in training."""
    prev = cur = None
    for nxt in frames:
        if cur is not None:
            yield prev, cur, nxt
        prev, cur = (nxt if cur is None else cur), nxt
    if cur is not None:
        yield prev, cur, cur


def model_detections(frames, path):
    """Ball from the trained net: at most one (x, y, score) per frame."""
    import torch
    from train_ball import THRESH, H, W, BallNet, peak, to_input

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BallNet().to(dev).eval()
    net.load_state_dict(torch.load(path, map_location=dev))
    for p, c, n in triplets(frames):
        x = torch.from_numpy(to_input([p, c, n]))[None].to(dev).float() / 255
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            px, py, score = peak(net(x)[0], c.shape[1] / W, c.shape[0] / H)
        yield c, [(px, py, score)] if score > THRESH else []


def drop_static(dets_per_frame, cell=16, max_frac=0.01):
    """Remove detections in grid cells that fire too often (dartboard flicker, cup, shoes).

    A moving ball crosses a cell in a frame or two; clutter sits there for hundreds.
    """
    counts = {}
    for dets in dets_per_frame:
        for c in {(int(x) // cell, int(y) // cell) for x, y, _ in dets}:
            counts[c] = counts.get(c, 0) + 1
    limit = max(3, max_frac * len(dets_per_frame))
    return [[d for d in dets if counts[(int(d[0]) // cell, int(d[1]) // cell)] <= limit] for dets in dets_per_frame]


def track(dets_per_frame):
    """Greedy nearest-neighbour linking -> list of tracks, each a list of (frame, x, y).

    ponytail: greedy gating, swap for a Kalman filter if tracks jump to paddles/shoes.
    """
    tracks, cur, last = [], [], None
    for f, dets in enumerate(dets_per_frame):
        if cur and f - cur[-1][0] > MAX_GAP:
            tracks.append(cur)
            cur, last = [], None
        if not dets:
            continue
        if last is None:
            x, y, _ = max(dets, key=lambda d: d[2])
        else:
            limit = MAX_JUMP_PX * (f - cur[-1][0])
            x, y, _ = min(dets, key=lambda d: np.hypot(d[0] - last[0], d[1] - last[1]))
            if np.hypot(x - last[0], y - last[1]) > limit:
                continue
        cur.append((f, x, y))
        last = (x, y)
    if cur:
        tracks.append(cur)
    return [t for t in tracks if len(t) >= 5]


def bounces(tr, H, kink_px=6):
    """Frames where the path kinks sharply upward on screen, on the table.

    Gravity only ever curves the path down in the image (~2 px/frame^2 here); a sharp
    upward kink is a bounce or a paddle hit, and paddle hits happen off the table.
    kink_px is a calibration knob.
    """
    out = []
    for i in range(1, len(tr) - 1):
        (f0, _, y0), (f, x, y), (f2, _, y2) = tr[i - 1], tr[i], tr[i + 1]
        if f - f0 == 1 and f2 - f == 1 and y2 - 2 * y + y0 < -kink_px:
            X, Y = cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0, 0]
            # Skip the band around the net: the net clamps/tape jitter and fake kinks there.
            if 0 <= X <= TABLE_LEN_M and 0 <= Y <= TABLE_WID_M and abs(X - TABLE_LEN_M / 2) > 0.2:
                out.append((f, X, Y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--debug", help="write annotated video here")
    ap.add_argument("--model", help="trained ball_net.pt; default is the colour detector")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = read_frames(cap)
    detections = model_detections(frames, a.model) if a.model else color_detections(frames)
    frames_dets, writer = [], None
    for frame, dets in detections:
        frames_dets.append(dets)
        if a.debug:
            writer = writer or cv2.VideoWriter(a.debug, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.shape[1::-1])
            frame = frame.copy()  # the model still reads this frame as the next triplet's "previous"
            cv2.polylines(frame, [TABLE_PX.astype(int)], True, (0, 255, 0), 1)
            for x, y, _ in dets:
                cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), 2)
            writer.write(frame)
    if writer:
        writer.release()

    H = cv2.getPerspectiveTransform(TABLE_PX, TABLE_M)
    frames_dets = drop_static(frames_dets)
    tracks = track(frames_dets)
    with open("track.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "frame", "t_s", "x_px", "y_px"])
        for ti, tr in enumerate(tracks):
            w.writerows([ti, f, round(f / fps, 3), round(x, 1), round(y, 1)] for f, x, y in tr)

    speeds = []
    with open("speeds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "t_s", "from_m", "to_m", "dist_m", "dt_s", "speed_kmh"])
        for ti, tr in enumerate(tracks):
            b = bounces(tr, H)
            for (f0, x0, y0), (f1, x1, y1) in zip(b, b[1:]):
                dist, dt = np.hypot(x1 - x0, y1 - y0), (f1 - f0) / fps
                # ponytail: horizontal speed only (table plane); ignores vertical component.
                kmh = dist / dt * 3.6
                # A real rally bounces once per side: consecutive bounces straddle the net.
                if (x0 - TABLE_LEN_M / 2) * (x1 - TABLE_LEN_M / 2) < 0:
                    speeds.append(kmh)
                    w.writerow([ti, round(f0 / fps, 2), f"{x0:.2f},{y0:.2f}", f"{x1:.2f},{y1:.2f}",
                                round(dist, 2), round(dt, 3), round(kmh, 1)])

    n_det = sum(bool(d) for d in frames_dets)
    print(f"{len(frames_dets)} frames, ball candidates in {n_det}, {len(tracks)} tracks, {len(speeds)} bounce pairs")
    if speeds:
        print(f"speed km/h: median {np.median(speeds):.1f}, p90 {np.percentile(speeds, 90):.1f}, max {max(speeds):.1f}")


def _selfcheck():
    # Straight-line ball + one decoy blob far away: linker must follow the ball.
    dets = [[(10 + 20 * f, 100, 20), (900, 600, 50)] if f else [(10, 100, 20)] for f in range(10)]
    tr = track(dets)
    assert len(tr) == 1 and all(abs(p[1] - (10 + 20 * p[0])) < 1e-6 for p in tr[0]), tr
    # A blob stuck in one spot for every frame is clutter, not the ball.
    clutter = [[(500, 500, 10), (10 + 20 * f, 100, 20)] for f in range(400)]
    assert all(len(d) == 1 and d[0][0] != 500 for d in drop_static(clutter))
    # Ball falls (y grows, gravity curve +2) to the table centre, then rebounds -> exactly one bounce.
    cx, cy = TABLE_PX.mean(axis=0)
    ys = [cy - 10 * (5 - f) + (5 - f) ** 2 if f <= 5 else cy - 12 * (f - 5) + (f - 5) ** 2 for f in range(11)]
    b = bounces([(f, cx, y) for f, y in enumerate(ys)], cv2.getPerspectiveTransform(TABLE_PX, TABLE_M))
    assert [p[0] for p in b] == [5], b
    # Model sees (prev, cur, next) for every frame, edges padded.
    assert list(triplets(iter([1, 2, 3]))) == [(1, 1, 2), (1, 2, 3), (2, 3, 3)]
    assert list(triplets(iter([7]))) == [(7, 7, 7)]
    # Homography maps corners to metres.
    H = cv2.getPerspectiveTransform(TABLE_PX, TABLE_M)
    assert np.allclose(cv2.perspectiveTransform(TABLE_PX[None], H)[0], TABLE_M, atol=1e-3)


if __name__ == "__main__":
    _selfcheck()
    main()
