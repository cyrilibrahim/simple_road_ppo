# Simple Road Scenario

## Installation

You'll need git lfs: https://github.com/git-lfs/git-lfs/wiki/Installation

```
git clone https://github.com/Unity-Technologies/ml-agents/
cd ml-agents/python
pip install .

git clone https://github.com/cyrilibrahim/simple_road_ppo/
cd simple_road_ppo
git lfs pull
chmod u+x circuit_linux/circuit_linux.x86_64
```

## Run on Slurm

```
sinter -c 2 --mem=16000 --gres=gpu --reservation=res_stretch
LD_LIBRARY_PATH=/Tmp/lisa/os_v5/lib/glx:$LD_LIBRARY_PATH xvfb-run -n $SLURM_JOB_ID -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python main.py
```
