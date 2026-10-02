#!/usr/bin/env bash
# Debug video for one recording with one model version, e.g. `./debug.sh 1948 3`:
# runs videos/1948.mp4 through models/ball_net_v3.pt and writes everything to runs/v3_1948/
# (debug.mp4 playable in VLC, plus track.csv, speeds.csv, detections.csv, rejected.csv).
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -ne 2 ]; then
    echo "usage: $0 VIDEO_NUMBER MODEL_VERSION    e.g. $0 1948 3" >&2
    exit 1
fi
video="videos/$1.mp4"
model="models/ball_net_v$2.pt"
out="runs/v${2}_$1"
[ -f "$video" ] || { echo "no such video: $video" >&2; exit 1; }
[ -f "$model" ] || { echo "no such model: $model (have: $(ls models))" >&2; exit 1; }

mkdir -p "$out"
cd "$out"
uv run ../../ball_speed.py "../../$video" --model "../../$model" --debug dbg.mp4
# OpenCV writes mp4v, which this VLC can't play: re-encode to H.264.
ffmpeg -v error -y -i dbg.mp4 -c:v libx264 -crf 20 -pix_fmt yuv420p debug.mp4
rm dbg.mp4
echo "debug video: $out/debug.mp4"
