"""uv run --with-requirements requirements.txt --with httpx python test_app.py"""
import os
import tempfile

import cv2
import numpy as np

os.environ["DATA_DIR"] = tempfile.mkdtemp()
from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

src = os.path.join(os.environ["DATA_DIR"], "src.mp4")
w = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
for i in range(10):
    w.write(np.full((48, 64, 3), i * 20, np.uint8))
w.release()

c = TestClient(app)
with open(src, "rb") as fh:
    assert c.post("/api/videos", files={"file": ("my clip.mp4", fh)}).json() == {"name": "my_clip.mp4"}
assert c.post("/api/videos", files={"file": ("x.txt", b"hi")}).status_code == 400
assert c.get("/api/videos").json() == [{"name": "my_clip.mp4", "frames": 10, "labeled": 0}]
assert c.get("/api/videos/my_clip.mp4/frames/3.jpg").headers["content-type"] == "image/jpeg"
assert c.get("/api/videos/my_clip.mp4/frames/99.jpg").status_code == 404
assert c.get("/api/videos/..%2Flabels.db/labels").status_code in (200, 404)  # no crash
assert c.put("/api/videos/..%2Flabels.db/labels/1", json={"visible": False}).status_code == 404
assert c.put("/api/videos/my_clip.mp4/labels/2", json={"visible": True, "x": 10.26, "y": 5}).status_code == 200
assert c.put("/api/videos/my_clip.mp4/labels/3", json={"visible": False, "x": 1, "y": 1}).status_code == 200
assert c.put("/api/videos/my_clip.mp4/labels/4", json={"visible": True}).status_code == 400
assert c.get("/api/labels.csv").text.splitlines() == [
    "video,frame,visible,x,y", "videos/my_clip.mp4,2,1,10.3,5.0", "videos/my_clip.mp4,3,0,,"]
assert c.get("/api/videos/my_clip.mp4/random").json()["frame"] not in (2, 3)
c.delete("/api/videos/my_clip.mp4/labels/2")
assert list(c.get("/api/videos/my_clip.mp4/labels").json()) == ["3"]
print("ok")
