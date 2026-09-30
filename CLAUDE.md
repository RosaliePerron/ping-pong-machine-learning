# Ball speed project

## Before training a new model

`train_ball.py` overwrites `models/ball_net.pt` during training. Before every training run, back up the current model to the next unused version number, so each version can still be compared afterwards:

```sh
n=1; while [ -e models/ball_net_v$n.pt ]; do n=$((n+1)); done
cmp -s models/ball_net.pt models/ball_net_v$((n-1)).pt || cp models/ball_net.pt models/ball_net_v$n.pt
```

The `cmp` skips the copy when `ball_net.pt` is already saved as the latest version.
