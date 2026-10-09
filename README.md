# Ball speed

Measures ping-pong ball speed from video: detect the ball → track it → map bounces onto the table → bounce-to-bounce speed.

| File | What it does |
|---|---|
| `ball_speed.py` | Runs one video through a detector (colour by default, or a trained model with `--model`) and writes `track.csv`, `speeds.csv`, `detections.csv`, `rejected.csv` to the current directory, plus an annotated video with `--debug`. |
| `train_ball.py` | Trains the ball detector (a small U-Net) on `labels.csv` and writes the best model to `models/ball_net.pt`. |
| `label_ball.py` | Desktop tool for clicking the ball in frames. Writes `labels.csv`. |
| `labeler/` | Web version of the same labelling tool, so several people can label at once. |
| `debug.sh` | Makes a debug video for one video with one model version. |

Videos go in `videos/` and are named by number (`videos/1948.mp4`). They are git-ignored. Scripts run with [uv](https://docs.astral.sh/uv/), which installs each script's dependencies.

## Training a new model

1. **Back up the current model.** `train_ball.py` overwrites `models/ball_net.pt`, so first copy it to the next free version number:

   ```sh
   n=1; while [ -e models/ball_net_v$n.pt ]; do n=$((n+1)); done
   cmp -s models/ball_net.pt models/ball_net_v$((n-1)).pt || cp models/ball_net.pt models/ball_net_v$n.pt
   ```

   If `ball_net.pt` is already saved as the latest version, the copy is skipped.

2. **Get the labels into `labels.csv`.** The CLI labeller writes there directly. If labels were made in the web labeller, download them from its **Download labels.csv** link and merge them in. Its video paths are already `videos/<name>`.

3. **Train:**

   ```sh
   uv run train_ball.py            # --epochs 150 --batch 4 --labels labels.csv by default
   ```

   Every 5th run of consecutive labelled frames is held out for validation. Every 10 epochs the script prints validation precision, recall and F1, and saves `models/ball_net.pt` whenever F1 improves. Use `--batch 4` on a 4 GB GPU; batch 8 runs out of memory.

4. **Save it as a version** (run the backup command from step 1 again) and compare it to the previous versions with debug videos (see below).

## Generating a debug video

```sh
./debug.sh VIDEO_NUMBER MODEL_VERSION     # e.g. ./debug.sh 1948 3
```

This runs `videos/1948.mp4` through `models/ball_net_v3.pt` and writes everything to `runs/v3_1948/`:

- `debug.mp4`: annotated video (re-encoded to H.264 so VLC can play it). Needs `ffmpeg`.
- `track.csv`, `speeds.csv`, `detections.csv`, `rejected.csv`: the run's data. The labelling modes below read these files.

To test the model you just trained before giving it a version number, call `ball_speed.py` directly:

```sh
mkdir -p runs/new_1948 && cd runs/new_1948
uv run ../../ball_speed.py ../../videos/1948.mp4 --model ../../models/ball_net.pt --debug debug.mp4
```

## Labelling

Each label is one frame: `video,frame,visible,x,y`. It records either "ball at (x, y)" or "no ball in this frame". There are three ways to choose which frames to label. The CLI and the web app choose exactly the same frames for the same arguments, because the web app calls the functions in `label_ball.py`.

### 1. Random runs of consecutive frames (`plan`)

The starting set. The tool picks `--runs` random stretches of `--run-len` consecutive frames across the given videos. The order of the videos matters because it seeds the random pick.

```sh
uv run label_ball.py videos/1912.mp4 videos/1913.mp4 videos/1914.mp4 --runs 40 --run-len 10
```

Many of these frames fall between rallies, so this mode mostly teaches the model what "no ball" looks like.

### 2. Where the model lost the ball (`--from-run`): missed detections

Uses a `ball_speed.py` run of the same video. The tool picks:

- the first frames after a track ends when the next track starts soon after (the ball was lost mid-rally),
- frames around suspicious bounce pairs (too short, too fast or too slow), including the middle of the pair, where a bounce may have been missed.

```sh
uv run label_ball.py videos/1902.mp4 --from-run runs/v3_1902
```

### 3. Detections that are probably not the ball (`--false-positives`)

Uses a `ball_speed.py --model` run (it needs `detections.csv` and `rejected.csv`). It samples in turn from three groups, keeping picked frames at least 5 apart:

- **untracked**: detections the tracker dropped (single hits, tracks that are too short, static clutter),
- **borderline**: tracked detections with a score of 0.7 or less,
- **rejected**: bounce candidates that failed the physics checks.

```sh
uv run label_ball.py videos/1936.mp4 --false-positives runs/v3_1936 --max-frames 150
```

The model's guess is shown as a yellow circle: **Enter** = it is the ball, **n** = not the ball / no ball, **click** = the ball is somewhere else.

### Keys (CLI)

Left click = ball here · `n` or right click = ball not visible · `b` = back one frame · `q` = quit. Each label is saved as soon as you make it. Run the same command again to pick up where you stopped. To zoom, use the toolbar magnifier, and turn it off before clicking.

### Web labeller

```sh
cd labeler && docker compose up -d --build    # http://localhost:8000
```

It serves the videos from `videos/` and lists the `runs/` folders for modes 2 and 3. Labels are stored in `labeler/data/labels.db`, with the name of the person who labelled each frame. Get them as a CSV with **Download labels.csv**.
