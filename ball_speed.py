# /// script
# dependencies = ["opencv-python-headless", "numpy", "torch"]
# ///
"""Ping-pong ball speed from 1912.mp4: detect -> track -> table homography -> bounce-to-bounce speed.

uv run ball_speed.py 1912.mp4 [--debug out.mp4] [--model ball_net.pt]

Without --model the ball is found by colour + motion; with it, by the net from train_ball.py.
"""
import argparse
import csv
import sys

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

# Bounce detection: a sharp upward kink finds candidates, a local flight fit checks and times them.
# Calibration knobs.
KINK_PX = 6  # upward bend (px/frame^2) that makes a bounce candidate; gravity alone bends ~+1-2 downward
FIT_WINDOW = 6  # frames of flight fitted on each side of a candidate
FIT_TOL_PX = 5.0  # min outlier tolerance; grows to 30% of the ball's per-frame screen motion when fast
GRAVITY_SLACK = 0.2  # px/frame^2 a flight may bend upward on screen (perspective)
BOUNCE_DT_S = (0.15, 1.2)  # plausible time between bounces on opposite sides of the net


def detect(frame, fg):
    """Moving orange blobs -> list of (x, y, area)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI) & fg
    mask = cv2.dilate(mask, None)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    return [(*cents[i], stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MIN_AREA <= stats[i, cv2.CC_STAT_AREA] <= MAX_AREA]


def read_frames(cap, stage):
    """Yield frames, printing `progress STAGE n/total` to stderr (\r: one line in a terminal, lines for the labeler)."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step, n = max(1, total // 100), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print(file=sys.stderr)
            return
        n += 1
        if n % step == 0:
            print(f"progress {stage} {n}/{total}", end="\r", file=sys.stderr, flush=True)
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


def _fit(pts):
    """x(f), y(f) polynomials in frame number f: quadratic once there are 4+ points."""
    f = [p[0] for p in pts]
    deg = min(2, len(pts) - 2)
    return np.polyfit(f, [p[1] for p in pts], deg), np.polyfit(f, [p[2] for p in pts], deg)


def _miss(fit, p):
    return np.hypot(np.polyval(fit[0], p[0]) - p[1], np.polyval(fit[1], p[0]) - p[2])


def robust_fit(pts, min_pts=3):
    """Fit one short flight, dropping the worst point while it misses by more than the tolerance.

    Between contacts only gravity acts, so a short stretch of the image path is close to a parabola.
    The tolerance grows with the ball's screen speed: near the camera it moves (and smears) more.
    ponytail: image-space parabola, fine over FIT_WINDOW frames; fit in 3D through H for long flights.
    """
    steps = [np.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]) for a, b in zip(pts, pts[1:])]
    tol = max(FIT_TOL_PX, 0.3 * np.median(steps)) if steps else FIT_TOL_PX
    pts = list(pts)
    while len(pts) >= min_pts:
        fit = _fit(pts)
        worst = max(pts, key=lambda p: _miss(fit, p))
        if _miss(fit, worst) <= tol:
            return fit
        pts.remove(worst)
    return None


def to_table(H, x, y):
    return cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0, 0]


def on_table(X, Y):
    # Skip the band around the net: the net clamps/tape jitter and fake kinks there.
    return 0 <= X <= TABLE_LEN_M and 0 <= Y <= TABLE_WID_M and abs(X - TABLE_LEN_M / 2) > 0.2


def bounces(tr, H, rejected=None):
    """Bounces in one track -> [(frame, X, Y)] in table metres; frame is fractional when refined.

    Candidates: frames where the path kinks sharply upward on screen (gravity only bends it down).
    Each candidate is then checked with a flight fitted on each side, leaving the kink point out:
    - falling on screen into it and rising out of it (a lone wild point between two halves of one
      smooth flight fails this: both halves fall, or both rise),
    - gravity bending both halves down (y'' > 0 on screen),
    - same left-right direction on screen before and after (a paddle hit reverses it),
    and timed where the two fits cross, which is between frames. Candidates with too few points
    to fit keep the plain kink result, so this never loses bounces the kink rule would find
    except by rejecting them. Rejected candidates are appended to `rejected` as (frame, x, y, reason).
    """
    out = []
    # Off-table candidates are dropped anyway; labelling them only clutters the debug video.
    reject = ((lambda why: on_table(*to_table(H, x, y)) and rejected.append((f, x, y, why)))
              if rejected is not None else (lambda why: None))
    # Paddle hits (screen x velocity reverses, judged on 3-step medians so one wild point can't fake it):
    # fit windows stop at them, or a hit ~0.1-0.2 s after the bounce bends the "after" parabola upward.
    vx = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(tr, tr[1:])]
    hits = [tr[j][0] for j in range(3, len(tr) - 3) if np.median(vx[j - 3 : j]) * np.median(vx[j : j + 3]) < 0]
    for i in range(1, len(tr) - 1):
        (f0, _, y0), (f, x, y), (f2, _, y2) = tr[i - 1], tr[i], tr[i + 1]
        if not (f - f0 == 1 and f2 - f == 1 and y2 - 2 * y + y0 < -KINK_PX):
            continue
        if out and f - out[-1][0] < 4:  # one bounce often kinks on two neighbouring frames
            continue
        # A reversal at the candidate itself is for the paddle-hit check below, not a window edge.
        lo = max([h for h in hits if h < f - 2], default=-1)
        hi = min([h for h in hits if h > f + 2], default=float("inf"))
        before = [p for p in tr[max(0, i - FIT_WINDOW) : i] if f - p[0] <= FIT_WINDOW and p[0] > lo]
        after = [p for p in tr[i + 1 : i + 1 + FIT_WINDOW] if p[0] - f <= FIT_WINDOW and p[0] < hi]
        fa = robust_fit(before) if len(before) >= 3 else None
        fb = robust_fit(after) if len(after) >= 3 else None
        if fa is None or fb is None:
            X, Y = to_table(H, x, y)
            if on_table(X, Y):
                out.append((f, X, Y))
            continue
        (xa, ya), (xb, yb) = fa, fb
        if (len(ya) == 3 and ya[0] < -GRAVITY_SLACK) or (len(yb) == 3 and yb[0] < -GRAVITY_SLACK):
            reject("bends up")
            continue
        roots = [r.real for r in np.roots(np.polysub(ya, yb)) if abs(r.imag) < 1e-9 and f - 1.5 <= r.real <= f + 1.5]
        t = min(roots, key=lambda r: abs(r - f), default=float(f))
        if not np.polyval(np.polyder(ya), t) > 0 > np.polyval(np.polyder(yb), t):
            reject("no fall->rise")
            continue
        # Direction from screen x, not through H: H is only right ON the table and the ball is in the
        # air either side. The table runs left-right on screen and height mostly moves the ball
        # up-down, so a paddle hit flips the sign of x velocity.
        if np.polyval(np.polyder(xa), t) * np.polyval(np.polyder(xb), t) <= 0:
            reject("paddle hit")
            continue
        X, Y = to_table(H, (np.polyval(xa, t) + np.polyval(xb, t)) / 2, (np.polyval(ya, t) + np.polyval(yb, t)) / 2)
        if on_table(X, Y):
            out.append((t, X, Y))
    return out


def write_debug(video, out, fps, frames_dets, tracks, bnc, pairs, rejected, H):
    """Annotated copy of the video: what was detected, tracked and decided, and why."""
    to_px = lambda X, Y: tuple(int(v) for v in cv2.perspectiveTransform(np.float32([[[X, Y]]]), np.linalg.inv(H))[0, 0])
    pos = {f: (int(x), int(y)) for tr in tracks for f, x, y in tr}
    events = {}  # frame -> [(kind, pixel, text)], each shown for a moment after it happens

    def show(f, kind, px, text, n):
        for g in range(int(f), int(f) + n):
            events.setdefault(g, []).append((kind, px, text))

    for f, X, Y, _ in bnc:
        show(f, "bounce", to_px(X, Y), "", 20)
    for f, X, Y, kmh in pairs:
        show(f, "speed", to_px(X, Y), f"{kmh:.1f} km/h", 30)
    for f, x, y, why in rejected:
        show(f, "rejected", (int(x), int(y)), why, 20)
    colour = {"bounce": (255, 255, 0), "speed": (255, 255, 255), "rejected": (255, 0, 255)}
    legend = [("detection", (0, 0, 255)), ("track (last 10 frames)", (0, 255, 255)),
              ("bounce", colour["bounce"]), ("rejected bounce candidate + reason", colour["rejected"])]

    cap, writer = cv2.VideoCapture(video), None
    for n, frame in enumerate(read_frames(cap, "draw")):
        writer = writer or cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.shape[1::-1])
        cv2.polylines(frame, [TABLE_PX.astype(int)], True, (0, 255, 0), 1)
        trail = [pos[k] for k in range(n - 10, n + 1) if k in pos]
        if len(trail) > 1:
            cv2.polylines(frame, [np.int32(trail)], False, (0, 255, 255), 1)
        for x, y, *_ in frames_dets[n] if n < len(frames_dets) else []:
            cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), 2)
        for kind, px, text in events.get(n, []):
            if kind == "bounce":
                cv2.circle(frame, px, 12, colour[kind], 2)
            elif kind == "rejected":
                cv2.drawMarker(frame, px, colour[kind], cv2.MARKER_TILTED_CROSS, 16, 2)
            if text:
                cv2.putText(frame, text, (px[0] + 14, px[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour[kind], 2)
        for i, (label, c) in enumerate(legend):
            cv2.putText(frame, label, (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        cv2.putText(frame, f"{n / fps:6.2f} s", (frame.shape[1] - 110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame)
    if writer:
        writer.release()


def load_tracks(path):
    tracks = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            tracks.setdefault(int(r["track"]), []).append((int(r["frame"]), float(r["x_px"]), float(r["y_px"])))
    return [tracks[k] for k in sorted(tracks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--debug", help="write annotated video here")
    ap.add_argument("--model", help="trained ball_net.pt; default is the colour detector")
    ap.add_argument("--tracks", help="reuse a previous run's track.csv instead of detecting (fast)")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    H = cv2.getPerspectiveTransform(TABLE_PX, TABLE_M)
    if a.tracks:
        tracks = load_tracks(a.tracks)
        frames_dets = [[] for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]
        for tr in tracks:
            for f, x, y in tr:
                if f < len(frames_dets):
                    frames_dets[f].append((x, y, 0))
    else:
        frames = read_frames(cap, "detect")
        detections = model_detections(frames, a.model) if a.model else color_detections(frames)
        frames_dets = [dets for _, dets in detections]
        # Every raw detection, before clutter filtering and tracking: label_ball.py --false-positives
        # samples the ones that got thrown away. score is the model's confidence (colour mode: blob area).
        with open("detections.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "x_px", "y_px", "score"])
            w.writerows([f, round(x, 1), round(y, 1), round(float(s), 3)] for f, dets in enumerate(frames_dets) for x, y, s in dets)
        frames_dets = drop_static(frames_dets)
        tracks = track(frames_dets)
    with open("track.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "frame", "t_s", "x_px", "y_px"])
        for ti, tr in enumerate(tracks):
            w.writerows([ti, f, round(f / fps, 3), round(x, 1), round(y, 1)] for f, x, y in tr)

    speeds, pairs, rejected = [], [], []
    # Pair consecutive bounces across tracks too: a rally split by a short detection gap still counts,
    # and the physics rules below reject pairs that skipped a bounce.
    b = sorted((f, X, Y, ti) for ti, tr in enumerate(tracks) for f, X, Y in bounces(tr, H, rejected))
    with open("speeds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "t_s", "from_m", "to_m", "dist_m", "dt_s", "speed_kmh"])
        for (f0, x0, y0, ti), (f1, x1, y1, _) in zip(b, b[1:]):
            dist, dt = np.hypot(x1 - x0, y1 - y0), (f1 - f0) / fps
            # ponytail: horizontal speed only (table plane); ignores vertical component.
            kmh = dist / dt * 3.6
            # A real rally bounces once per side: consecutive bounces straddle the net.
            if (x0 - TABLE_LEN_M / 2) * (x1 - TABLE_LEN_M / 2) < 0 and BOUNCE_DT_S[0] <= dt <= BOUNCE_DT_S[1]:
                speeds.append(kmh)
                pairs.append((f1, x1, y1, kmh))
                w.writerow([ti, round(f0 / fps, 2), f"{x0:.2f},{y0:.2f}", f"{x1:.2f},{y1:.2f}",
                            round(dist, 2), round(dt, 3), round(kmh, 1)])

    with open("rejected.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x_px", "y_px", "reason"])
        w.writerows([f, round(x, 1), round(y, 1), why] for f, x, y, why in rejected)

    n_det = sum(bool(d) for d in frames_dets)
    print(f"{len(frames_dets)} frames, ball candidates in {n_det}, {len(tracks)} tracks, "
          f"{len(b)} bounces ({len(rejected)} candidates rejected), {len(speeds)} bounce pairs")
    if speeds:
        print(f"speed km/h: median {np.median(speeds):.1f}, p90 {np.percentile(speeds, 90):.1f}, max {max(speeds):.1f}")
    if a.debug:
        write_debug(a.video, a.debug, fps, frames_dets, tracks, b, pairs, rejected, H)


def _selfcheck():
    # Straight-line ball + one decoy blob far away: linker must follow the ball.
    dets = [[(10 + 20 * f, 100, 20), (900, 600, 50)] if f else [(10, 100, 20)] for f in range(10)]
    tr = track(dets)
    assert len(tr) == 1 and all(abs(p[1] - (10 + 20 * p[0])) < 1e-6 for p in tr[0]), tr
    # A blob stuck in one spot for every frame is clutter, not the ball.
    clutter = [[(500, 500, 10), (10 + 20 * f, 100, 20)] for f in range(400)]
    assert all(len(d) == 1 and d[0][0] != 500 for d in drop_static(clutter))
    # Ball falls onto the table at (0.7 m, 0.65 m) at frame 10.4 (between frames) and rebounds, with one
    # wild detection mid-flight that kinks upward (a fake candidate) -> one bounce, timed and placed
    # from the fits, the wild point neither a bounce itself nor spoiling the fit.
    H = cv2.getPerspectiveTransform(TABLE_PX, TABLE_M)
    bx, by = cv2.perspectiveTransform(np.float32([[[0.7, 0.65]]]), np.linalg.inv(H))[0, 0]
    tb = 10.4
    arc = [(f, bx + 6 * (f - tb), by + 0.8 * (f - tb) ** 2 + (8 if f < tb else -8) * (f - tb)) for f in range(21)]
    arc[5] = (5, arc[5][1], arc[5][2] - 30)
    b = bounces(arc, H)
    assert len(b) == 1 and abs(b[0][0] - tb) < 0.05 and np.allclose(b[0][1:], (0.7, 0.65), atol=0.02), b
    # One smooth flight with a wild detection is not a bounce (the old kink rule called it one).
    flight = [(f, bx + 6 * (f - tb), by + 0.8 * (f - tb) ** 2 - 8 * (f - tb)) for f in range(21)]
    flight[10] = (10, flight[10][1], flight[10][2] - 30)
    assert bounces(flight, H) == [], bounces(flight, H)
    # Same rebound, but the ball comes back the way it came: a paddle hit, not a bounce.
    hit = [(f, bx + 6 * abs(f - tb), y) for f, _, y in arc[:5] + arc[6:]]
    assert bounces(hit, H) == [], bounces(hit, H)
    # Model sees (prev, cur, next) for every frame, edges padded.
    assert list(triplets(iter([1, 2, 3]))) == [(1, 1, 2), (1, 2, 3), (2, 3, 3)]
    assert list(triplets(iter([7]))) == [(7, 7, 7)]
    # Homography maps corners to metres.
    H = cv2.getPerspectiveTransform(TABLE_PX, TABLE_M)
    assert np.allclose(cv2.perspectiveTransform(TABLE_PX[None], H)[0], TABLE_M, atol=1e-3)


if __name__ == "__main__":
    _selfcheck()
    main()
