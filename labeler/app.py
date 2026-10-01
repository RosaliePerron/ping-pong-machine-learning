"""Frame labelling API: upload videos, click the ball in frames, export labels.csv (video,frame,visible,x,y)."""
import csv
import io
import os
import re
import sqlite3
import sys
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

DATA = Path(os.environ.get("DATA_DIR", "/data"))
VIDEOS = DATA / "videos"
VIDEOS.mkdir(parents=True, exist_ok=True)
EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
REPO = Path(os.environ.get("REPO_DIR", "/repo"))
RUNS = REPO / "runs"
sys.path.insert(0, str(REPO))
import label_ball  # noqa: E402  the CLI's own frame pickers, so web and CLI choose the same frames

db = sqlite3.connect(DATA / "labels.db", check_same_thread=False, isolation_level=None)
db.execute("PRAGMA journal_mode=WAL")
db.execute("""CREATE TABLE IF NOT EXISTS labels (
    video TEXT, frame INTEGER, visible INTEGER, x REAL, y REAL, labeler TEXT,
    PRIMARY KEY (video, frame))""")

app = FastAPI()


def video_path(name: str) -> Path:
    p = VIDEOS / name
    if p.parent != VIDEOS or not p.is_file():  # blocks ../ traversal
        raise HTTPException(404, "no such video")
    return p


def frame_count(p: Path) -> int:
    return int(cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FRAME_COUNT))


@app.get("/api/videos")
def list_videos():
    done = dict(db.execute("SELECT video, COUNT(*) FROM labels GROUP BY video").fetchall())
    return [{"name": p.name, "frames": frame_count(p), "labeled": done.get(p.name, 0)}
            for p in sorted(VIDEOS.iterdir()) if p.suffix.lower() in EXTS]


@app.post("/api/videos")
def upload(file: UploadFile):
    name = re.sub(r"[^\w.-]", "_", Path(file.filename or "").name)
    if Path(name).suffix.lower() not in EXTS:
        raise HTTPException(400, f"expected one of {sorted(EXTS)}")
    dest = VIDEOS / name
    if dest.exists():
        raise HTTPException(409, f"{name} already exists")
    with dest.open("wb") as fh:
        while chunk := file.file.read(1 << 20):
            fh.write(chunk)
    if frame_count(dest) <= 0:
        dest.unlink()
        raise HTTPException(400, "could not read video")
    return {"name": name}


@app.get("/api/videos/{name}/frames/{frame}.jpg")
def frame_jpg(name: str, frame: int):
    # ponytail: opens + seeks per request, fine for a few labellers; keep decoders open if it gets slow
    cap = cv2.VideoCapture(str(video_path(name)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    if not ok:
        raise HTTPException(404, "no such frame")
    return Response(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(),
                    media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/videos/{name}/labels")
def video_labels(name: str):
    rows = db.execute("SELECT frame, visible, x, y, labeler FROM labels WHERE video=?", (name,))
    return {f: {"visible": v, "x": x, "y": y, "labeler": who} for f, v, x, y, who in rows}


# files each label_ball.py mode reads from a ball_speed.py run folder
MODE_FILES = {"from-run": {"track.csv", "speeds.csv"},
              "false-positives": {"track.csv", "detections.csv", "rejected.csv"}}


def run_modes(d: Path) -> list[str]:
    files = {f.name for f in d.iterdir()}
    return [m for m, need in MODE_FILES.items() if need <= files]


@app.get("/api/runs")
def list_runs():
    dirs = sorted(d for d in RUNS.iterdir() if d.is_dir()) if RUNS.is_dir() else []
    return [{"name": d.name, "modes": run_modes(d)} for d in dirs]


@app.get("/api/queue")
def queue(mode: str, videos: list[str] = Query(), run: str = "",
          runs: int = Query(40, ge=1, le=1000), run_len: int = Query(10, ge=1, le=1000),
          max_frames: int = Query(150, ge=1, le=5000)):
    """Frames to label, picked exactly like `label_ball.py` with the same arguments.

    plan: label_ball.py VIDEOS --runs --run-len (video order matters, it seeds the pick)
    from-run: label_ball.py VIDEO --from-run runs/RUN
    false-positives: label_ball.py VIDEO --false-positives runs/RUN --max-frames
    """
    paths = [str(video_path(v)) for v in videos]
    guesses, counts = {}, None
    if mode == "plan":
        if min(frame_count(Path(p)) for p in paths) <= run_len:
            raise HTTPException(400, "run_len is longer than a video")
        items = label_ball.plan(paths, runs, run_len)
    elif mode in MODE_FILES:
        if len(paths) != 1:
            raise HTTPException(400, f"{mode} takes the one video that run was made from")
        run_dir = RUNS / run
        if run_dir.parent != RUNS or not run_dir.is_dir() or mode not in run_modes(run_dir):
            raise HTTPException(404, f"no run folder {run!r} with {sorted(MODE_FILES[mode])}")
        if mode == "from-run":
            items = label_ball.hard_frames(paths[0], str(run_dir))
        else:
            items, guesses, counts = label_ball.suspect_detections(paths[0], str(run_dir), max_frames)
    else:
        raise HTTPException(400, "mode is plan, from-run or false-positives")
    return {"items": [{"video": Path(v).name, "frame": f, "guess": guesses.get((v, f))} for v, f in items],
            "counts": counts}


class Label(BaseModel):
    visible: bool
    x: float | None = None
    y: float | None = None
    labeler: str = ""


@app.put("/api/videos/{name}/labels/{frame}")
def put_label(name: str, frame: int, label: Label):
    video_path(name)
    if label.visible and (label.x is None or label.y is None):
        raise HTTPException(400, "visible labels need x and y")
    x, y = (label.x, label.y) if label.visible else (None, None)
    db.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?)",
               (name, frame, int(label.visible), x, y, label.labeler[:50]))
    return {"ok": True}


@app.delete("/api/videos/{name}/labels/{frame}")
def delete_label(name: str, frame: int):
    db.execute("DELETE FROM labels WHERE video=? AND frame=?", (name, frame))
    return {"ok": True}


@app.get("/api/labels.csv")
def export():
    """Same columns as the repo's labels.csv, video paths as videos/<name>."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["video", "frame", "visible", "x", "y"])
    for v, f, vis, x, y in db.execute("SELECT video, frame, visible, x, y FROM labels ORDER BY video, frame"):
        w.writerow([f"videos/{v}", f, vis, "" if x is None else round(x, 1), "" if y is None else round(y, 1)])
    return Response(out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=labels.csv"})


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")
