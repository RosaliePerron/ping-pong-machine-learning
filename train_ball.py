# /// script
# dependencies = ["torch", "opencv-python-headless", "numpy"]
# ///
"""Train a small TrackNet-style ball detector on labels.csv (from label_ball.py).

Input: previous, current and next frame at 640x360, stacked into 9 channels.
Output: heatmap of where the ball is in the current frame. Best model -> models/ball_net.pt.

uv run train_ball.py [--epochs 150]
"""
import argparse
import csv
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

W, H = 640, 360  # model input size: half of 1280x720
SIGMA = 1.5  # target blob radius, model pixels
THRESH = 0.5  # heatmap peak above this = "ball visible". Calibration knob: trades misses vs false alarms.
HIT_PX = 8  # prediction within this many ORIGINAL pixels of the label counts as a hit


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class BallNet(nn.Module):
    """Small U-Net: 9-channel frame triplet -> ball heatmap logits (B, H, W)."""

    def __init__(self, c=(32, 64, 128, 256)):
        super().__init__()
        self.down = nn.ModuleList([block(9, c[0]), block(c[0], c[1]), block(c[1], c[2])])
        self.mid = block(c[2], c[3])
        self.up = nn.ModuleList([block(c[3] + c[2], c[2]), block(c[2] + c[1], c[1]), block(c[1] + c[0], c[0])])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for d in self.down:
            x = d(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for u in self.up:
            skip = skips.pop()
            x = u(torch.cat([F.interpolate(x, size=skip.shape[2:]), skip], 1))
        return self.out(x)[:, 0]


def to_input(frames_bgr):
    """3 BGR frames (any size) -> (9, H, W) uint8 array."""
    return np.concatenate([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames_bgr], 2).transpose(2, 0, 1)


def peak(logits, sx, sy):
    """Heatmap logits (H, W) -> (x, y, score) in original pixels."""
    p = torch.sigmoid(logits.float())
    i = int(p.argmax())
    return (i % W + 0.5) * sx - 0.5, (i // W + 0.5) * sy - 0.5, float(p.flatten()[i])


def heatmap(vis, x, y, sx, sy):
    if not vis:
        return np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (x + 0.5) / sx - 0.5, (y + 0.5) / sy - 0.5
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * SIGMA ** 2)).astype(np.float32)


def load_data(path):
    with open(path, newline="") as fh:
        rows = [(r["video"], int(r["frame"]), int(r["visible"]), float(r["x"] or 0), float(r["y"] or 0))
                for r in csv.DictReader(fh)]
    rows.sort()
    frames, scale = {}, {}
    for v in sorted({r[0] for r in rows}):
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scale[v] = (cap.get(cv2.CAP_PROP_FRAME_WIDTH) / W, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / H)
        last = None
        for g in sorted({min(max(g, 0), n - 1) for r in rows if r[0] == v for g in (r[1] - 1, r[1], r[1] + 1)}):
            if g != last:  # seek only on jumps, read sequentially inside a run (seeking is slow)
                cap.set(cv2.CAP_PROP_POS_FRAMES, g)
            ok, frames[(v, g)] = cap.read()
            assert ok, (v, g)
            last = g + 1
    x = np.stack([to_input([frames.get((v, g), frames[(v, f)]) for g in (f - 1, f, f + 1)]) for v, f, *_ in rows])
    y = np.stack([heatmap(vis, px, py, *scale[v]) for v, f, vis, px, py in rows])
    # Run id: consecutive frames are near-duplicates, so split train/val by run, never by frame.
    run, runs = -1, []
    for i, (v, f, *_) in enumerate(rows):
        run += i == 0 or (v, f - 1) != rows[i - 1][:2]
        runs.append(run)
    return rows, x, y, np.array(runs), scale


def evaluate(model, rows, x, idx, scale, dev):
    tp = fp = fn = 0
    model.eval()
    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        for i in idx:
            v, _, vis, lx, ly = rows[i]
            px, py, score = peak(model(torch.from_numpy(x[i : i + 1]).to(dev).float() / 255)[0], *scale[v])
            hit = vis and score > THRESH and np.hypot(px - lx, py - ly) <= HIT_PX
            tp += hit
            fp += score > THRESH and not hit
            fn += vis and not hit
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4)  # 8 runs out of memory on a 4 GB GPU
    a = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, x, y, runs, scale = load_data(a.labels)
    val_mask = runs % 5 == 0  # every 5th run held out
    tr, va = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    print(f"{len(rows)} labelled frames ({sum(r[2] for r in rows)} with ball), "
          f"train {len(tr)} / val {len(va)} in {runs.max() + 1} runs, device {dev}")

    model = BallNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = a.epochs * -(-len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    pos_weight = torch.tensor(200.0, device=dev)  # ball is ~10 of 230k pixels; without this it predicts "no ball"
    best, t0 = -1.0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        np.random.shuffle(tr)
        total = 0.0
        for b in range(0, len(tr), a.batch):
            idx = tr[b : b + a.batch]
            xb = torch.from_numpy(x[idx]).to(dev).float() / 255
            yb = torch.from_numpy(y[idx]).to(dev)
            if random.random() < 0.5:  # mirror
                xb, yb = xb.flip(-1), yb.flip(-1)
            # Lighting jitter, same for all 3 frames of a sample: the three videos are lit differently.
            gain = torch.empty(len(idx), 1, 1, 1, device=dev).uniform_(0.5, 1.5)
            xb = (xb * gain + torch.empty_like(gain).uniform_(-0.1, 0.1)).clamp(0, 1)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        left = (time.time() - t0) / ep * (a.epochs - ep)
        print(f"\repoch {ep}/{a.epochs}  loss {total / len(tr):.4f}  ~{left / 60:.0f} min left ", end="", flush=True)
        if ep % 10 == 0 or ep == a.epochs:
            print()
            prec, rec = evaluate(model, rows, x, va, scale, dev)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            mark = ""
            if f1 > best:
                best, mark = f1, "  <- saved"
                torch.save(model.state_dict(), "models/ball_net.pt")
            print(f"  val precision {prec:.2f}  recall {rec:.2f}  F1 {f1:.2f}{mark}", flush=True)
    print(f"best val F1 {best:.2f} -> models/ball_net.pt")


def _selfcheck():
    # Heatmap peak decodes back to the labelled pixel (within one original pixel).
    hm = torch.from_numpy(heatmap(1, 701.0, 333.0, 2.0, 2.0))
    px, py, _ = peak(torch.logit(hm.clamp(1e-6, 1 - 1e-6)), 2.0, 2.0)
    assert abs(px - 701) <= 1 and abs(py - 333) <= 1, (px, py)
    assert BallNet()(torch.zeros(1, 9, H, W)).shape == (1, H, W)


if __name__ == "__main__":
    _selfcheck()
    main()
