"""Frame labelling API: upload videos, click the ball in frames, export labels.csv (video,frame,visible,x,y)."""
import csv
import io
import os
import filecmp
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
from collections import deque
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
MODELS = REPO / "models"
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
def list_videos(page: int = Query(1, ge=1), per_page: int = Query(25, ge=1, le=200)):
    """One page of videos; frame counts (opening each file) only for that page, which is what made it slow."""
    names = sorted((p.name for p in VIDEOS.iterdir() if p.suffix.lower() in EXTS), reverse=True,
                   key=lambda n: (int(m[0]) if (m := re.match(r"\d+", n)) else -1, n))  # newest number first
    done = dict(db.execute("SELECT video, COUNT(*) FROM labels GROUP BY video").fetchall())
    rows = [{"name": n, "frames": frame_count(VIDEOS / n), "labeled": done.get(n, 0)}
            for n in names[(page - 1) * per_page:page * per_page]]
    return {"videos": rows, "total": len(names), "page": page, "per_page": per_page}


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


def labels_csv() -> str:
    """Same columns as the repo's labels.csv, video paths as videos/<name>."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["video", "frame", "visible", "x", "y"])
    for v, f, vis, x, y in db.execute("SELECT video, frame, visible, x, y FROM labels ORDER BY video, frame"):
        w.writerow([f"videos/{v}", f, vis, "" if x is None else round(x, 1), "" if y is None else round(y, 1)])
    return out.getvalue()


@app.get("/api/labels.csv")
def export():
    return Response(labels_csv(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=labels.csv"})


@app.get("/api/models")
def list_models() -> list[int]:
    """Saved model versions: models/ball_net_vN.pt -> N."""
    return sorted(int(m[1]) for p in MODELS.glob("ball_net_v*.pt") if (m := re.fullmatch(r"ball_net_v(\d+)\.pt", p.name)))


jobs: dict[str, dict] = {}  # run name -> status. ponytail: in memory, lost on restart (finished videos stay in runs/)


class DebugJob(BaseModel):
    video: str
    version: int


@app.post("/api/debug")
def start_debug(job: DebugJob):
    """debug.sh VIDEO VERSION in the background, into runs/vVERSION_VIDEO/; poll GET /api/debug/{run}."""
    video, model = video_path(job.video), MODELS / f"ball_net_v{job.version}.pt"
    if not model.is_file():
        raise HTTPException(404, f"no model v{job.version}")
    run = f"v{job.version}_{video.stem}"
    if jobs.get(run, {}).get("state") == "running":
        raise HTTPException(409, f"{run} is already running")
    jobs.pop(run, None)  # a re-run moves to the end of the list
    jobs[run] = {"state": "running", "stage": "starting", "done": 0, "total": 0}
    threading.Thread(target=make_debug, args=(run, video, model), daemon=True).start()
    return {"run": run}


def make_debug(run: str, video: Path, model: Path):
    job, out, log = jobs[run], RUNS / run, deque(maxlen=20)
    try:
        out.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen([sys.executable, str(REPO / "ball_speed.py"), str(video), "--model", str(model),
                              "--debug", "dbg.mp4"], cwd=out, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:  # text mode turns ball_speed.py's \r progress into lines
            if m := re.fullmatch(r"progress (\w+) (\d+)/(\d+)\n", line):
                job.update(stage=m[1], done=int(m[2]), total=int(m[3]))
            elif line.strip():
                log.append(line.rstrip())
        if p.wait():
            raise RuntimeError("\n".join(log))
        job.update(stage="encode", done=0, total=0)
        # OpenCV writes mp4v, which VLC and browsers can't play: re-encode to H.264, like debug.sh
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "dbg.mp4", "-c:v", "libx264", "-crf", "20",
                            "-pix_fmt", "yuv420p", "debug.mp4"], cwd=out, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        (out / "dbg.mp4").unlink()
        job.update(state="done", log="\n".join(log))
    except Exception as e:
        job.update(state="error", error=str(e) or repr(e))


def run_dir(run: str) -> Path:
    d = RUNS / run
    if d.parent != RUNS:  # blocks ../ traversal
        raise HTTPException(404, "no such run")
    return d


@app.get("/api/debug")
def list_debug():
    """Every job since the server started, oldest first: run name -> status."""
    return dict(jobs)


@app.get("/api/debug/{run}")
def debug_status(run: str):
    if run in jobs:
        return jobs[run]
    if (run_dir(run) / "debug.mp4").is_file():
        return {"state": "done"}
    raise HTTPException(404, "no such job")


@app.get("/api/debug/{run}/debug.mp4")
def debug_video(run: str):
    p = run_dir(run) / "debug.mp4"
    if not p.is_file():
        raise HTTPException(404, "no debug video yet")
    return FileResponse(p, media_type="video/mp4", filename=f"{run}.mp4")


def save_model(always: bool) -> int | None:
    """models/ball_net.pt -> ball_net_vN.pt with the next N, unless (not always) it is already the latest version.
    Returns the version ball_net.pt is saved as, None without a ball_net.pt."""
    src, n = MODELS / "ball_net.pt", max(list_models(), default=0)
    if not src.is_file():
        return None
    if not always and n and filecmp.cmp(src, MODELS / f"ball_net_v{n}.pt", shallow=False):
        return n
    shutil.copy2(src, MODELS / f"ball_net_v{n + 1}.pt")
    return n + 1


training: dict = {"state": "idle"}  # ponytail: in memory like jobs; a restart mid-training loses the status
train_lock = threading.Lock()


@app.post("/api/train")
def start_train():
    """Back up the model and labels.csv, replace labels.csv with the database's labels, train_ball.py, save as a
    new version. In the background; poll GET /api/train."""
    if not train_lock.acquire(blocking=False):
        raise HTTPException(409, "a training is already running")
    training.clear()
    training.update(state="running", stage="backup", done=0, total=0)
    threading.Thread(target=train, daemon=True).start()
    return training


@app.get("/api/train")
def train_status():
    return training


def train():
    job, log, prev = training, deque(maxlen=20), None
    try:
        prev = save_model(always=False)  # CLAUDE.md: back up ball_net.pt before every training run
        # labels.csv trained the previous version: tag it with that number. An existing labels_vN.csv is
        # the original from an earlier failed attempt, whose labels.csv is already a database export.
        labels, old = REPO / "labels.csv", REPO / f"labels_v{prev or max(list_models(), default=0)}.csv"
        if labels.is_file() and not old.exists():
            shutil.copy2(labels, old)
        labels.write_text(labels_csv())
        job.update(stage="load", backup=old.name)
        p = subprocess.Popen([sys.executable, "-u", str(REPO / "train_ball.py")], cwd=REPO, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:  # text mode turns train_ball.py's \r epoch line into lines
            if m := re.match(r"epoch (\d+)/(\d+)", line):
                job.update(stage="train", done=int(m[1]), total=int(m[2]))
            elif line.strip():
                log.append(line.rstrip())
                if "val precision" in line:
                    job["val"] = line.strip()
        if p.wait():
            raise RuntimeError("\n".join(log))
        job.update(state="done", version=save_model(always=True), log="\n".join(log))
    except Exception as e:
        if prev:  # train_ball.py may have overwritten ball_net.pt before failing: put the saved one back
            shutil.copy2(MODELS / f"ball_net_v{prev}.pt", MODELS / "ball_net.pt")
        job.update(state="error", error=str(e) or repr(e))
    finally:
        train_lock.release()


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")
