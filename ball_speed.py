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


def paddle_hits(tr):
    """Frames where screen x velocity reverses, judged on 3-step medians so one wild point can't fake it.
    One hit flips it on 2-3 neighbouring frames."""
    vx = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(tr, tr[1:])]
    return [tr[j][0] for j in range(3, len(tr) - 3) if np.median(vx[j - 3 : j]) * np.median(vx[j : j + 3]) < 0]


def hit_times(tr):
    """Paddle hits, one per hit, timed between frames where the image paths before and after it meet."""
    raw = paddle_hits(tr)
    out = []
    for h in [grp.mean() for grp in np.split(np.float64(raw), np.flatnonzero(np.diff(raw) > 3) + 1) if len(grp)]:
        before = [p for p in tr if h - FIT_WINDOW <= p[0] < h]
        after = [p for p in tr if h < p[0] <= h + FIT_WINDOW]
        fa = robust_fit(before) if len(before) >= 3 else None
        fb = robust_fit(after) if len(after) >= 3 else None
        if fa is not None and fb is not None:
            h = min(np.linspace(h - 1.5, h + 1.5, 61), key=lambda s: np.hypot(
                np.polyval(fa[0], s) - np.polyval(fb[0], s), np.polyval(fa[1], s) - np.polyval(fb[1], s)))
        out.append(h)
    return out


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
    # Fit windows stop at paddle hits, or a hit ~0.1-0.2 s after the bounce bends the "after" parabola upward.
    hits = paddle_hits(tr)
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


def camera():
    """(K, R, t) of the camera, world = table metres with z up.

    Assumes the principal point at the centre of the 1280x720 frame TABLE_PX was clicked on, square pixels and
    no lens distortion, so the focal length is the only unknown intrinsic: 4 corners, 8 equations, 7 unknowns.
    ponytail: 1.5 px of click error moves it ~35 px (camera ~10 cm); click the net post tops too if 3D looks off.
    """
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    guess = np.float64([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])
    obj = np.float32([[X, Y, 0] for X, Y in TABLE_M])
    _, K, _, (rv,), (tv,) = cv2.calibrateCamera([obj], [TABLE_PX], (1280, 720), guess, None, flags=flags)
    R, t = cv2.Rodrigues(rv)[0], tv.ravel()
    if (-R.T @ t)[2] < 0:  # the corners don't say which side is up; the camera is above the table
        R[:, 2] *= -1
    return K, R, t


GRAVITY = 9.81  # m/s^2
JOIN_W = ANCHOR_W = 10  # how hard flights meet at paddle hits / touch the table at bounces; detections weigh 1 per metre
TABLE_H_M = 0.76  # table top above the floor


def flights(tracks, H, fps):
    """The ball in 3D -> [(t0, t1, p, v)]: from t0 to t1 s it is at p + v (t - t0) + gravity, table metres, z up.

    One camera can't see depth, but between contacts the ball flies a parabola under known gravity, a bounce
    pins it to the table and at a paddle hit the flights before and after meet. Each track's flights are fitted
    together by linear least squares on how far the ball passes from the camera ray through each detection.
    Left out (guessed() places those detections): flights that don't touch a bounce or meet one that does
    (serve tosses, pick-ups: nothing fixes their depth), and flights that aren't a parabola (rolling, carried,
    a contact the tracker missed) or go through the floor.
    ponytail: no drag or spin, fine at these 4-8 m/s; add drag if fast shots come out too high.
    """
    K, R, t = camera()
    P, Kinv = np.hstack([R, t[:, None]]), np.linalg.inv(K)  # normalised: ray misses come out in metres at the ball
    g = np.float64([0, 0, -GRAVITY])
    out = []
    for tr in tracks:
        edges = sorted([(f, (X, Y)) for f, X, Y in bounces(tr, H)] + [(h, "hit") for h in hit_times(tr)],
                       key=lambda e: e[0])
        edges = [(tr[0][0] - 0.5, None)] + edges + [(tr[-1][0] + 0.5, None)]
        segs = [(e0, e1, [p for p in tr if e0[0] < p[0] < e1[0]]) for e0, e1 in zip(edges, edges[1:])]
        n, rows, rhs, owner = len(segs), [], [], []

        def eq(coefs, val, w, k=None):  # coefs: {flight: 6 coefficients on (p, v)}
            r = np.zeros(6 * n)
            for j, c in coefs.items():
                r[6 * j : 6 * j + 6] = c
            rows.append(w * r)
            rhs.append(w * val)
            owner.append(k)

        for k, ((f0, c0), (f1, c1), pts) in enumerate(segs):
            if len(pts) < 4:
                continue
            for f, u, v in pts:
                tau = (f - f0) / fps
                xn, yn, _ = Kinv @ [u, v, 1]
                for a in (P[0] - xn * P[2], P[1] - yn * P[2]):  # a . (x, y, z, 1) = 0 on the ray
                    eq({k: np.r_[a[:3], a[:3] * tau]}, -a[3] - a[:3] @ g * tau * tau / 2, 1, k)
            T = (f1 - f0) / fps
            for c, tau in ((c0, 0), (c1, T)):
                if isinstance(c, tuple):
                    for i, want in enumerate((*c, 0)):
                        eq({k: np.eye(6)[i] + tau * np.eye(6)[3 + i]}, want - g[i] * tau * tau / 2, ANCHOR_W)
            if c1 == "hit" and k + 1 < n and len(segs[k + 1][2]) >= 4:
                for i in range(3):
                    eq({k: np.eye(6)[i] + T * np.eye(6)[3 + i], k + 1: -np.eye(6)[i]}, -g[i] * T * T / 2, JOIN_W)
        if not rows:
            continue
        A, b = np.array(rows), np.array(rhs)
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        miss, owner = np.abs(A @ sol - b), np.array(owner, dtype=object)
        bounced = [len(pts) >= 4 and (isinstance(c0, tuple) or isinstance(c1, tuple)) for (_, c0), (_, c1), pts in segs]
        for k, ((f0, c0), (f1, c1), pts) in enumerate(segs):
            p, v = sol[6 * k : 6 * k + 3], sol[6 * k + 3 : 6 * k + 6]
            T = (f1 - f0) / fps
            z = [p[2] + v[2] * s + g[2] * s * s / 2 for s in np.linspace(0, T, 10)]
            pinned = bounced[k] or (c0 == "hit" and bounced[k - 1]) or (c1 == "hit" and k + 1 < n and bounced[k + 1])
            fits = len(pts) >= 4 and np.median(miss[owner == k]) < 0.05
            if pinned and fits and -TABLE_H_M - 0.05 < min(z) and max(z) < 3:
                out.append((f0 / fps, f1 / fps, p, v))
    return out


def guessed(tracks, fl, fps):
    """Detections no flight covers -> [(t, x, y, z)]: on the camera ray through each, at a guessed depth.

    The depth (distance from the camera) is interpolated in time between the track's flights on either side, so
    the ball doesn't jump where it leaves or rejoins one; before the first or after the last it holds that
    flight's depth. In a track with no flight at all the ball goes on the table's centre line, where rallies fly.
    """
    K, R, t = camera()
    C, M = -R.T @ t, R.T @ np.linalg.inv(K)  # camera centre; pixel -> ray direction, table metres
    g = np.float64([0, 0, -GRAVITY])
    out = []
    for tr in tracks:
        mine = [f for f in fl if tr[0][0] <= (f[0] + f[1]) / 2 * fps <= tr[-1][0]]  # tracks never overlap in time
        ends = sorted((s, np.linalg.norm(p + v * (s - t0) + g * (s - t0) ** 2 / 2 - C))
                      for t0, t1, p, v in mine for s in (t0, t1))
        for f, u, v in tr:
            s = f / fps
            if any(t0 <= s <= t1 for t0, t1, *_ in mine):
                continue
            d = M @ [u, v, 1]
            d /= np.linalg.norm(d)
            if ends:  # np.interp holds the end values outside the flights
                depth = np.interp(s, [e[0] for e in ends], [e[1] for e in ends])
            elif d[1] < -0.1:  # ray heads away from the camera across the table's long axis
                depth = (TABLE_WID_M / 2 - C[1]) / d[1]
            else:
                depth = np.linalg.norm(C - (TABLE_LEN_M / 2, TABLE_WID_M / 2, 0))
            if d[2] < 0:  # not through the floor
                depth = min(depth, (-TABLE_H_M - C[2]) / d[2])
            out.append((s, *(C + depth * d)))
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
    # A rally filmed by camera(): bounce on the left half, the left player's hit, bounce on the right half.
    # flights() gets the 3D path back from the pixels alone.
    K, R, t = camera()
    assert (-R.T @ t)[2] > 1, "camera above the table"
    g = np.float64([0, 0, -GRAVITY])
    fly = lambda p, v, t0: lambda s: np.float64(p) + np.float64(v) * (s - t0) + g * (s - t0) ** 2 / 2
    after1 = fly([0.8, 0.5, 0], [-4, 0.1, 2.9], 0.5)
    back = fly(after1(0.85), [5, 0.1, 1.5], 0.85)
    tb = 0.85 + (1.5 + np.sqrt(1.5 ** 2 + 2 * GRAVITY * after1(0.85)[2])) / GRAVITY  # lands at x ~1.8
    legs = [(0, fly([0.8, 0.5, 0], [-5, -0.2, -3.2], 0.5)), (0.5, after1), (0.85, back), (tb, fly(back(tb), [4, 0.1, 2.8], tb))]
    truth = lambda s: [leg for t0, leg in legs if t0 <= s][-1](s)
    rally = []
    for f in range(48):
        u, v, w = K @ (R @ truth(f / 30) + t)
        rally.append((f, u / w, v / w))
    fl = flights([rally], H, 30)
    assert len(fl) == 4, fl
    err = [np.linalg.norm(p + v * (f / 30 - t0) + g * (f / 30 - t0) ** 2 / 2 - truth(f / 30))
           for f, *_ in rally for t0, t1, p, v in fl if t0 <= f / 30 <= t1]
    assert len(err) >= 47 and max(err) < 0.05, (len(err), max(err))
    # Detections no flight covers go on their camera ray: with the middle flight missing, at depths between its
    # neighbours' (within 8 cm of the truth here); with no flight at all, on the table's centre line.
    assert guessed([rally], fl, 30) == []
    gs = guessed([rally], fl[:1] + fl[2:], 30)
    assert len(gs) >= 10 and max(np.linalg.norm(np.array(xyz) - truth(s)) for s, *xyz in gs) < 0.1, gs
    for s, *xyz in gs:
        u, v, w = K @ (R @ np.array(xyz) + t)
        assert np.allclose((u / w, v / w), rally[round(s * 30)][1:]), (s, u / w, v / w)
    line = [(f, 500 + 8 * f, 200.0) for f in range(10)]
    assert np.allclose([y for _, _, y, _ in guessed([line], [], 30)], TABLE_WID_M / 2)


if __name__ == "__main__":
    _selfcheck()
    main()
