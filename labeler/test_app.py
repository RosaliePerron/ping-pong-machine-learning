"""uv run --with-requirements requirements.txt --with httpx --with numpy python test_app.py"""
import os
import shutil
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["DATA_DIR"] = tempfile.mkdtemp()
os.environ["REPO_DIR"] = repo = tempfile.mkdtemp()
shutil.copy(os.path.join(HERE, "..", "label_ball.py"), repo)
run = os.path.join(repo, "runs", "v1_my_clip")
os.makedirs(run)
for name, text in {"track.csv": "track,frame,t_s,x_px,y_px\n0,1,0,9,9\n1,6,0,9,9\n",
                   "speeds.csv": "track,t_s,from_m,to_m,dist_m,dt_s,speed_kmh\n",
                   "detections.csv": "frame,x_px,y_px,score\n1,9,9,0.5\n",
                   "rejected.csv": "frame,x_px,y_px,reason\n8,30,20,x\n"}.items():
    with open(os.path.join(run, name), "w") as fh:
        fh.write(text)
from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

src = os.path.join(os.environ["DATA_DIR"], "videos", "my_clip.mp4")
w = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
for i in range(10):
    w.write(np.full((48, 64, 3), i * 20, np.uint8))
w.release()

c = TestClient(app)
assert c.post("/api/videos").status_code == 405  # no uploads: videos come from the repo's videos/
assert c.get("/api/videos").json() == {
    "videos": [{"name": "my_clip.mp4", "frames": 10, "labeled": 0}], "total": 1, "page": 1, "per_page": 25}
extras = ("999.mp4", "1000.mp4", "1000b.mp4")
for extra in extras:  # paging + numeric descending order, no-number names last
    shutil.copy(src, os.path.join(os.environ["DATA_DIR"], "videos", extra))
p1 = c.get("/api/videos", params={"page": 1, "per_page": 3}).json()
p2 = c.get("/api/videos", params={"page": 2, "per_page": 3}).json()
assert [v["name"] for v in p1["videos"] + p2["videos"]] == ["1000b.mp4", "1000.mp4", "999.mp4", "my_clip.mp4"]
assert p2["total"] == 4, p2
assert c.get("/api/videos", params={"page": 0}).status_code == 422
for extra in extras:
    os.remove(os.path.join(os.environ["DATA_DIR"], "videos", extra))
assert c.get("/api/videos/my_clip.mp4/frames/3.jpg").headers["content-type"] == "image/jpeg"
assert c.get("/api/videos/my_clip.mp4/frames/99.jpg").status_code == 404
assert c.get("/api/videos/..%2Flabels.db/labels").status_code in (200, 404)  # no crash
assert c.put("/api/videos/..%2Flabels.db/labels/1", json={"visible": False}).status_code == 404
assert c.put("/api/videos/my_clip.mp4/labels/2", json={"visible": True, "x": 10.26, "y": 5}).status_code == 200
assert c.put("/api/videos/my_clip.mp4/labels/3", json={"visible": False, "x": 1, "y": 1}).status_code == 200
assert c.put("/api/videos/my_clip.mp4/labels/4", json={"visible": True}).status_code == 400
assert c.get("/api/labels.csv").text.splitlines() == [
    "video,frame,visible,x,y", "videos/my_clip.mp4,2,1,10.3,5.0", "videos/my_clip.mp4,3,0,,"]

# queues pick the same frames as label_ball.py
import label_ball  # noqa: E402

clip = os.path.join(os.environ["DATA_DIR"], "videos", "my_clip.mp4")
q = c.get("/api/queue", params={"mode": "plan", "videos": "my_clip.mp4", "runs": 3, "run_len": 4}).json()
assert [it["frame"] for it in q["items"]] == [f for _, f in label_ball.plan([clip], 3, 4)]
assert c.get("/api/queue", params={"mode": "plan", "videos": "my_clip.mp4", "run_len": 10}).status_code == 400
assert c.get("/api/runs").json() == [{"name": "v1_my_clip", "modes": ["from-run", "false-positives"],
                                      "files": ["detections.csv", "rejected.csv", "speeds.csv", "track.csv"]}]
q = c.get("/api/queue", params={"mode": "from-run", "videos": "my_clip.mp4", "run": "v1_my_clip"}).json()
assert [it["frame"] for it in q["items"]] == [2, 3, 4], q
q = c.get("/api/queue", params={"mode": "false-positives", "videos": "my_clip.mp4", "run": "v1_my_clip"}).json()
assert q["items"] == [{"video": "my_clip.mp4", "frame": 1, "guess": [9.0, 9.0]},
                      {"video": "my_clip.mp4", "frame": 8, "guess": [30.0, 20.0]}], q
assert q["counts"] == {"untracked": 0, "borderline": 1, "rejected": 1}
assert c.get("/api/queue", params={"mode": "from-run", "videos": "my_clip.mp4", "run": ".."}).status_code == 404
c.delete("/api/videos/my_clip.mp4/labels/2")
assert list(c.get("/api/videos/my_clip.mp4/labels").json()) == ["3"]

# debug video: a stand-in ball_speed.py (prints progress, copies the video), then the real ffmpeg
import time  # noqa: E402

os.makedirs(os.path.join(repo, "models"))
for v in (2, 10):
    open(os.path.join(repo, "models", f"ball_net_v{v}.pt"), "w").close()
open(os.path.join(repo, "models", "ball_net.pt"), "w").close()
with open(os.path.join(repo, "ball_speed.py"), "w") as fh:
    fh.write("import shutil, sys\nfor n in (5, 10): print(f'progress detect {n}/10', end='\\r', file=sys.stderr)\n"
             "print('10 frames')\nshutil.copy(sys.argv[1], sys.argv[sys.argv.index('--debug') + 1])\n")
assert c.get("/api/models").json() == [2, 10]
assert c.post("/api/debug", json={"video": "my_clip.mp4", "version": 3}).status_code == 404
assert c.post("/api/debug", json={"video": "../labels.db", "version": 2}).status_code == 404
assert c.post("/api/debug", json={"video": "my_clip.mp4", "version": 2}).json() == {"run": "v2_my_clip"}
for _ in range(100):
    if (job := c.get("/api/debug/v2_my_clip").json())["state"] != "running":
        break
    time.sleep(0.1)
assert job["state"] == "done", job
assert job["log"] == "10 frames", job
r = c.get("/api/debug/v2_my_clip/debug.mp4")
assert r.headers["content-disposition"] == 'attachment; filename="v2_my_clip.mp4"', r.headers
assert cv2.VideoCapture(os.path.join(repo, "runs", "v2_my_clip", "debug.mp4")).get(cv2.CAP_PROP_FRAME_COUNT) == 10
assert not os.path.exists(os.path.join(repo, "runs", "v2_my_clip", "dbg.mp4"))
assert c.get("/api/debug/v1_my_clip/debug.mp4").status_code == 404
# run page: plays debug.mp4 inline, seekable
r = c.get("/api/runs/v2_my_clip/debug.mp4", headers={"Range": "bytes=0-9"})
assert r.status_code == 206 and len(r.content) == 10, r
assert r.headers["content-disposition"].startswith("inline"), r.headers
assert c.get("/api/runs/v1_my_clip/track.csv").text.startswith("track,frame")
for bad in ("v1_my_clip/nope.csv", "v1_my_clip/..%2F..%2Fmodels%2Fball_net.pt", "..%2Fmodels/ball_net.pt", "v1_my_clip/."):
    assert c.get(f"/api/runs/{bad}").status_code == 404, bad
assert c.get("/api/debug/..%2F..%2Fetc").status_code == 404
print("ok")
