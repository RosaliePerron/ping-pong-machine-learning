"""Frame labelling API: upload videos, click the ball in frames, export labels.csv (video,frame,visible,x,y)."""
import csv
import io
import os
import random
import re
import sqlite3
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

DATA = Path(os.environ.get("DATA_DIR", "/data"))
VIDEOS = DATA / "videos"
VIDEOS.mkdir(parents=True, exist_ok=True)
EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

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


@app.get("/api/videos/{name}/random")
def random_unlabeled(name: str):
    """A random unlabelled frame, so several people on one video don't collide."""
    n = frame_count(video_path(name))
    done = {f for (f,) in db.execute("SELECT frame FROM labels WHERE video=?", (name,))}
    left = [f for f in range(n) if f not in done]
    if not left:
        raise HTTPException(404, "every frame is labelled")
    return {"frame": random.choice(left)}


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
